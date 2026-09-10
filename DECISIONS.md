# Decisions — Pathfinder Agent

Every non-obvious choice, the alternatives, and why they lost.

---

## D-01 · Algorithms written out, not `networkx`

**Chose:** Kahn, Tarjan, Dijkstra and A* implemented directly.

**Why:** networkx does each of these in one line, and "I called `nx.topological_sort`" is
not an answer to "what happens when the graph has a cycle?" This module is the part that
has to be defensible line by line.

**Cost, stated honestly:** networkx is better tested than this code. That trade is
acceptable for a project whose purpose is demonstrating the algorithms, and would be the
wrong trade in production.

## D-02 · Everything iterative, never recursive

**Why:** real dependency chains reach depths that overflow CPython's default 1,000-frame
recursion limit. A resolver that crashes on a deep graph fails exactly when the problem is
hardest. Recursive Tarjan is shorter and prettier and dies on real input.

**Enforced by a test** that drives a 4,000-node chain through every algorithm.

## D-03 · Kahn for topological sort, not DFS post-order

**Chose:** Kahn's algorithm with an in-degree queue.

**Why:**
- It detects cycles naturally. If the queue empties with nodes remaining, the remainder
  *is* the cyclic part — which is what the caller needs to be told.
- It is iterative by construction.
- The frontier is explicit, so a deterministic tie-break is a one-line addition.

**Why a tie-break at all:** without it, output order depends on dict insertion order. A
resolver that produces a different valid answer each run cannot be tested or diffed.

**The bug this had.** The first version decremented the in-degree of each node's
*predecessors* rather than its *successors* — it walked the reversed graph. Every real
successor stayed above zero, the queue drained early, and the function reported a cycle on
a provably acyclic graph. It was caught because `detect_cycles` returned an empty list at
the same moment: **two algorithms disagreeing about the same graph is the cheapest bug
signal there is**, and it is why both are exposed rather than one.

## D-04 · Tarjan for cycles, not "DFS and remember the stack"

**Why:** Tarjan finds *all* strongly connected components in one pass. When a lockfile is
broken in several places at once you would rather see every cycle than the first one.

**Validated on real data:** it found a genuine circular dependency in the Apache Airflow
package family on the live PyPI graph — an 8-package cycle, not a synthetic test case.

## D-05 · Lazy-deletion heaps, not decrease-key

**Chose:** push duplicates, skip stale pops.

**Why:** Python's `heapq` has no decrease-key. The alternatives are an indexed heap or
tombstoning, both of which cost more code and more places to be subtly wrong than the
duplicate entries cost memory on a sparse graph.

**The precondition, stated:** Dijkstra requires non-negative weights. Every weight here is
an install cost, which cannot be negative. If that changed, this would silently return
wrong answers and Bellman-Ford would be the correct call.

## D-06 · Heuristics carry an admissibility proof *and* a checker

**Chose:** every heuristic states its argument in the docstring, and `verify_admissible()`
checks the claim against Dijkstra ground truth.

**Why the checker exists — and it earned its place immediately.** The depth heuristic's
first version returned `|level(goal) − level(n)|` with a confident proof attached. The
proof only covers the case where the goal is deeper. When the goal is *shallower* the
absolute difference over-estimates, which makes the heuristic inadmissible — and an
inadmissible heuristic does not error. **A* silently stops being optimal.** The checker
caught it; a reviewer reading the docstring very likely would not have.

**Fixed to** `max(0, level(goal) − level(n))`, with the argument rewritten to cover both
directions.

**And it turned out not to matter:** the corrected heuristic prunes *nothing* — identical
expansions to Dijkstra on every pair tested. The trivial constant heuristic prunes 8× on
two of them. The clever idea lost to the obvious one, which is reported rather than
buried.

## D-07 · Real PyPI data, extras excluded

**Chose:** live PyPI JSON API, `requires_dist` parsed with environment markers honoured.

**Why exclude extras:** `pytest; extra == "test"` is not installed by default. Including
optional dependencies roughly triples the graph with test and docs tooling that no user of
the package ever receives, which would make every depth number an artifact of the parser.

**Consequence, discovered by measuring:** an initial root set of ordinary libraries topped
out at **depth 4**. That is a real property of modern pip packages, and it was useless as a
sweep axis. The roots were widened to orchestration and ML stacks (`apache-airflow`, `dvc`,
`jupyterlab`, `langchain`, `mlflow`, `prefect`) to reach depth 9.

**BFS not DFS when truncating at `max_nodes`:** a depth-first walk that hits the cap leaves
one branch fully explored and the rest untouched — a badly skewed graph. BFS leaves a
uniform frontier.

## D-08 · Grade plans by their properties, not against one answer

**Chose:** check coverage, duplicates, invented packages, and ordering violations
separately.

**Why:** a dependency graph normally admits many valid topological orders. Comparing
against a single reference would mark correct plans wrong and turn the evaluation into a
measurement of agreement-with-my-tie-break rather than of correctness.

**Why four separate checks:** "wrong" is not actionable. "Missing 12 of 41 packages,
first: `anyio`" is. The failure-mode breakdown — 89% missing packages, 0% ordering
violations — is only possible because the checks are separate, and that breakdown is what
pointed at the real mechanism.

## D-09 · Two chunk kinds in the corpus

**Chose:** a *description* chunk and a *dependency* chunk per package.

**Why:** the agent asks two different questions — "what is this for?" and "what does it
pull in?" A single chunk blending both makes every embedding a mixture of two topics, and
both queries retrieve it equally badly. This is the most common RAG failure, and it is a
chunking decision, not a model decision.

## D-10 · Hybrid reranking, not a cross-encoder

**Chose:** dense score plus an exact package-name boost plus an inferred-kind boost.

**Why not a cross-encoder:** it would be a second neural model, a second download, and
tens of milliseconds per query, to reorder six candidates on a corpus with strong
structural signal that vector similarity throws away.

**Substring matching is deliberately avoided:** `click` would otherwise match
`clickhouse-driver` and quietly poison results. Names match on whole normalised tokens.

**The bug here.** Kind inference originally counted `{"what", "does", "for"}` as
description signals, so *"what does flask depend on?"* scored description 2 against
dependency 1 — the interrogatives outvoted the one word carrying the topic. Fixed by
removing question words from the sets entirely and weighting dependency terms 2×, because
they are specific ("transitive", "requires") where description terms are broad ("use",
"package").

## D-11 · Exact numpy search by default, pgvector for deployment

**Chose:** brute-force cosine over 822 unit vectors as the default; pgvector implemented
and schema-complete behind the same interface.

**Why exact is genuinely correct here, not a shortcut:** 822 × 384 ≈ 316k multiply-adds,
well under a millisecond in numpy. HNSW's advantage begins around 10⁵–10⁶ vectors. An
approximate index at this scale would add a dependency, a build step, a recall parameter to
tune and a source of nondeterminism — in exchange for being slower.

**Why pgvector exists anyway:** it is the deployment target and it proves the abstraction
is real. IVFFlat over HNSW because it builds far faster on a small corpus and its recall is
tunable at query time rather than baked in at build time.

## D-12 · flan-t5-**base**, after small failed outright

**Chose:** `google/flan-t5-base`, 512-token context, local, no key.

**flan-t5-small was tried first and rejected.** Asked to choose a tool it replied *"resolve
python package installation order"* — it echoed the task instead of acting on it. Every run
scored zero for reasons that had nothing to do with the variable under study, and **a model
that cannot do the task at depth 1 cannot measure degradation with depth.**

**Why a small context is a feature:** 512 tokens reaches the truncation cliff at a depth
this corpus actually contains. The same experiment on a 128k-context model would find the
cliff far later — which is the point being made, not a limitation of it.

**Greedy decoding, no sampling:** the evaluation compares plans across depths, and sampling
would add run-to-run variance indistinguishable from the effect being measured.

## D-13 · The LLM adapter measures and *reports* truncation

**Chose:** every backend exposes `context_limit`, a real `count_tokens`, and per-call
`truncated` / `tokens_dropped`.

**Why:** most agent frameworks hide truncation. That is precisely the bug worth measuring —
when a prompt loses its tail, the model answers confidently about information it never
received, and the failure presents as bad reasoning.

**Ironically, this instrumentation is what proved truncation was *not* the cause.** Because
the harness measured it honestly, it could report 0% rather than assuming the convenient
answer.

## D-14 · Memory evicts oldest-first, and never empties

**Why oldest-first:** in a resolution loop the most recent tool output is what the next
decision depends on. Dropping newest is the one policy guaranteed to remove exactly what is
needed.

**Why it keeps at least one observation:** a memory that empties itself is worse than one
that overruns — the agent would answer from nothing.

**Why memory gets a *fraction* of the context, not all of it:** the instructions, the tool
list and the model's own answer all need room. Budgeting 100% to memory guarantees the
instructions get truncated, which is the worst possible thing to lose.

## D-15 · `ToolResult` carries `summary` **and** `answer_text` — the headline decision

**Chose:** two renderings of the same information.

- `summary` — prose with a lead-in. Reads well in a planning trace.
- `answer_text` — the bare list, no framing.

**Why, measured:** handing an instruction-tuned seq2seq model *"Correct install order for
typer: annotated-doc, colorama, …"* and asking *"what is the correct install order for
typer?"* triggers **extractive QA**. The model returns the shortest span that answers the
question — the word `typer`. Measured at **0.12 validity**; the raw output for a
five-package closure was one word.

Removing the lead-in: **0.75 validity, 0.95 coverage.** Across the full sweep, overall
validity went **0.578 → 0.804** and coverage **0.873 → 0.954**.

**The generalisable lesson:** the format a tool returns is part of the agent's interface,
not a cosmetic detail. It deserves the same care as the tool's schema, and it is invisible
in every agent framework I know of.

**Guarded by a CI gate** that fails the build if overall validity drops below 0.70, because
this is exactly the kind of change a later refactor would undo without noticing.

## D-16 · Give the agent a tool that returns the exact answer

**Chose:** `correct_install_order` is available and returns ground truth.

**Why this is not cheating:** the evaluation measures whether the agent *uses its tools
well as the problem grows*, not whether a language model can do topological sort in its
head. An agent that calls this tool and copies the result should score 100%.

**That is what makes a failure interesting.** When it scores 12%, the answer was always
available, sitting in its context, well inside the budget — so every explanation involving
capability, context or reasoning is ruled out before the investigation starts.

## D-17 · Test three mechanisms against each other, not one

**Chose:** `eval/truncation.py` states M1/M2/M3 explicitly and falsifies each.

**Why:** "LLMs get unreliable on long problems" is a restatement, not an explanation. The
three mechanisms have *different fixes* — a bigger context, one config line, or a better
model — so prescribing the wrong one wastes the effort entirely.

Two of the three were falsified by direct intervention rather than by argument: a 4k
context rescues 0 of 19, and raising the output cap 320 → 1024 improved 0 of 10.

## D-18 · Expose the algorithm next to the agent in the UI

**Chose:** `/compare/{package}` returns both, and the UI shows them side by side with
omissions in red.

**Why:** most agent demos expose only the agent, which makes it impossible to see what the
agent costs you. Putting the exact answer beside it turns a demo into a measurement anyone
can rerun — including an interviewer.
