"""The agent: memory budgeting, tool dispatch, parsing, and the format finding.

These use the `echo` backend, which needs no model download. That is deliberate: these
tests check the HARNESS - budgets, parsing, eviction, dispatch - and mixing a real
model in would make them slow and flaky without testing anything extra. The model's
behaviour is measured by eval/, not asserted here.
"""

import pytest

from agent.llm import EchoLLM, get_llm
from agent.loop import PathfinderAgent, parse_package_list, parse_tool_call
from agent.memory import Memory
from agent.tools import Toolbox
from graph.algorithms import Graph


@pytest.fixture()
def graph():
    g = Graph()
    for a, b in [("app", "web"), ("app", "db"), ("web", "http"), ("db", "http"),
                 ("http", "sock")]:
        g.add_edge(a, b)
    return g


# -- memory ------------------------------------------------------------------

def test_memory_evicts_oldest_first_when_over_budget():
    """Oldest-first because in a resolution loop the most recent tool output is what
    the next decision depends on."""
    memory = Memory(budget_tokens=10, count_tokens=lambda t: len(t.split()))
    for i in range(6):
        memory.add(i, "tool", f"observation number {i} padding padding")
    assert memory.evicted, "nothing was evicted despite exceeding the budget"
    assert memory.evicted[0].step == 0
    assert memory.observations[-1].step == 5


def test_memory_never_evicts_its_last_observation():
    """A memory that empties itself is worse than one that overruns: the agent would
    answer from nothing."""
    memory = Memory(budget_tokens=1, count_tokens=lambda t: len(t.split()))
    memory.add(0, "tool", "a very long observation " * 20)
    assert len(memory.observations) == 1


def test_render_plain_drops_the_step_prefixes():
    """REGRESSION: with '[0] tool:' prefixes, flan-t5-base's most common answer was the
    literal string '[0]' - it copied the prefix instead of the content."""
    memory = Memory(budget_tokens=1000, count_tokens=lambda t: len(t.split()))
    memory.add(0, "correct_install_order", "Correct install order for x: a, b", "a, b")
    assert "[0]" in memory.render()
    assert "[0]" not in memory.render_plain()
    assert memory.render_plain() == "a, b"


# -- tools -------------------------------------------------------------------

def test_tools_return_real_algorithmic_answers(graph):
    toolbox = Toolbox(graph)
    assert "web" in toolbox.direct_dependencies("app").payload["dependencies"]
    assert toolbox.dependency_depth("app").payload["depth"] == 3
    assert toolbox.full_closure("app").payload["closure"] == sorted(graph.nodes)
    assert toolbox.correct_install_order("app").payload["order"][-1] == "app"


def test_a_tool_says_when_it_truncated(graph):
    """A tool that silently returns the first 20 of 200 items teaches the agent that
    20 is the answer."""
    big = Graph()
    for i in range(80):
        big.add_edge("root", f"dep{i}")
    result = Toolbox(big, max_items=10).full_closure("root")
    assert result.truncated
    assert "showing 10 of" in result.summary


def test_unknown_packages_fail_cleanly(graph):
    result = Toolbox(graph).direct_dependencies("nope")
    assert not result.ok and "not in the graph" in result.summary


def test_answer_text_has_no_lead_in(graph):
    """THE FINDING, pinned as a test. The lead-in 'Correct install order for X:' causes
    short-span extraction and costs 6.25x validity on flan-t5-base."""
    result = Toolbox(graph).correct_install_order("app")
    assert result.summary.startswith("Correct install order for app:")
    assert not result.answer_text.startswith("Correct install order")
    assert result.answer_text.split(", ")[-1] == "app"


def test_cycles_are_reported_by_the_tool():
    g = Graph()
    for a, b in [("a", "b"), ("b", "c"), ("c", "a")]:
        g.add_edge(a, b)
    assert Toolbox(g).find_cycles().payload["cycles"]


# -- parsing -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("correct_install_order(flask)", "correct_install_order"),
    ("I will call full_closure(app) next", "full_closure"),
    ("Tool: DIRECT_DEPENDENCIES(web)", "direct_dependencies"),
    ("let me use dependency_depth", "dependency_depth"),
])
def test_tool_calls_are_parsed_forgivingly(graph, text, expected):
    """Small models do not emit clean syntax. A strict parser would confound
    instruction-following with the context effect being measured."""
    registry = Toolbox(graph).registry()
    call = parse_tool_call(text, registry)
    assert call is not None and call[0] == expected


def test_nonsense_yields_no_tool_call(graph):
    assert parse_tool_call("the weather is nice today", Toolbox(graph).registry()) is None


def test_package_lists_are_filtered_against_the_real_package_set(graph):
    """A model asked for package names will happily produce prose; counting 'the' as a
    package would flatter every score."""
    parsed = parse_package_list("first install sock, then http and finally app",
                                set(graph.nodes))
    assert parsed == ["sock", "http", "app"]


def test_parsing_preserves_order_and_duplicates(graph):
    parsed = parse_package_list("sock, http, sock, app", set(graph.nodes))
    assert parsed == ["sock", "http", "sock", "app"]


# -- the loop ----------------------------------------------------------------

def test_the_agent_runs_end_to_end_on_the_echo_backend(graph):
    agent = PathfinderAgent(graph, EchoLLM(context_limit=400), Toolbox(graph))
    run = agent.run("app")
    assert run.steps
    assert run.context_limit == 400
    assert isinstance(run.grade.valid, bool)


def test_truncation_is_recorded_not_hidden(graph):
    """The whole point of the LLM adapter: a prompt that overruns the context must be
    RECORDED as truncated, not silently trimmed."""
    agent = PathfinderAgent(graph, EchoLLM(context_limit=12), Toolbox(graph))
    run = agent.run("app")
    assert run.any_truncation
    assert run.total_tokens_dropped > 0


def test_memory_budget_is_a_fraction_of_the_context(graph):
    """Budgeting 100% of the context to memory guarantees the instructions get
    truncated, which is the worst possible thing to lose."""
    agent = PathfinderAgent(graph, EchoLLM(context_limit=1000), Toolbox(graph))
    assert agent.memory_budget < 1000


def test_the_agent_stops_once_it_has_the_exact_answer(graph):
    """More tool calls after the answer is in memory can only push it out of budget."""
    agent = PathfinderAgent(graph, EchoLLM(context_limit=4000), Toolbox(graph),
                            max_steps=5)
    run = agent.run("app")
    assert run.tools_used.count("correct_install_order") <= 1
