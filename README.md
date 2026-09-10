# Pathfinder Agent

Graph search over **real PyPI dependency graphs**, with an LLM agent attempting the same
task and graded against the exact algorithm. Retrieval-augmented, pgvector-backed,
deployed.

**4,232 lines · 51 tests passing · 411 real packages, 989 edges · verified on CPython 3.13.9**

Every number below was measured and written to `outputs/`. Nothing is estimated.

---

## The finding

The project set out to test a specific, plausible claim:

> Agent plan quality collapses as dependency depth grows, and the cause is
> context-window truncation.

**Both halves are false, and the evidence says so unambiguously.**

Quality did fall with depth. But when the failures were attributed to a mechanism:

| candidate mechanism | share of failures |
|---|---:|
| M1 · input truncation | **0.0%** |
| M2 · output token limit | 21.1% |
| M3 · neither | 63.2% |
| M1 and M2 together | 15.8% |

And the falsification tests came back flat:

- **Would a bigger context have rescued them?** All 19 failing answer prompts fit inside
  512 tokens. A 4k context rescues **0 of 19**.
- **Was the output being cut off?** Raising `max_new_tokens` from 320 to 1024 gave
  **identical coverage on all 10 tested failures. 0 improved.**
- **Was it reasoning?** Across every failure in the sweep there were **zero ordering
  violations.** The model never once put a package before its dependency. 89% of failures
  were *missing packages*.

So: the information was present, the budget was sufficient, and the ordering was never
wrong. Something else was happening.

### What was actually happening

Looking at raw output on a package with a **five-package** closure:

```
TOOL SAID : Correct install order for jsonschema-specifications:
            attrs, rpds-py, typing-extensions, referencing, jsonschema-specifications
MODEL SAID: 'jsonschema-specifications'
```

And on `typer` (8 packages) → `'typer'`. On `transformers` (29 packages) → `'transformers'`.

flan-t5-base was treating the answer step as **extractive question answering**. Given a
passage containing *"Correct install order for typer: …"* and asked *"what is the correct
install order for typer?"*, it returned the shortest span that answers the question — the
package name itself. Not a context failure, not a size failure, not a reasoning failure.
A **format** failure, caused by the shape of the text the tool handed it.

### The experiment

Four answer-prompt formats, 8 packages, same model, same facts:

| variant | validity | coverage |
|---|---:|---:|
| names the package in the question *(original)* | 0.12 | 0.608 |
| "list every package name in the sentence above" | 0.12 | 0.796 |
| **bare list + "repeat the list above exactly"** | **0.75** | **0.950** |
| "extract the complete list, do not stop early" | 0.12 | 0.813 |

The winning variant **never mentions the package.** Naming it invites an extractive answer
whose shortest valid span is the name.

### The fix, and what it bought

The fix is in the **tool**, not the model and not the prompt. `ToolResult` now carries
`answer_text` — the bare list — alongside the prose `summary` used for planning traces.

Full sweep, before and after, identical tasks and seed:

| | before | after |
|---|---:|---:|
| overall validity | 0.578 | **0.804** |
| overall coverage | 0.873 | **0.954** |
| collapse depth *(first depth under 50% valid)* | 3 | **7** |
| deepest depth with any success | 6 | **9** |

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validity **before** | 1.00 | 0.83 | 0.33 | 1.00 | 0.83 | 0.33 | 0.00 | 0.00 | 0.00 |
| validity **after** | 0.88 | 1.00 | 1.00 | 0.88 | 0.75 | 0.86 | 0.33 | 0.25 | 0.50 |

Note the *before* row is not even monotonic — depth 3 sat at 0.33 while depth 4 hit 1.00.
That non-monotonicity was the first clue that depth was not the variable doing the work.

### And once the confound was gone, the real driver appeared

Correlations with plan validity, before and after the fix:

| driver | before | after |
|---|---:|---:|
| dependency depth | **−0.590** | −0.441 |
| closure size | −0.574 | −0.709 |
| prompt tokens | −0.510 | **−0.788** |

Before the fix, depth looked like the strongest predictor. After it, depth drops to third
and **prompt size becomes dominant.** Depth was a proxy: deeper packages have bigger
closures, and it was the size that mattered. The residual failures are now exactly the
packages with 90–109 dependencies, which genuinely do overflow a 512-token context —
so truncation is real, it just was never the *first* problem.

---

## Second finding: the clever heuristic lost to the trivial one

A* was given three admissible heuristics on the real graph. Node expansions:

| source → target | zero | depth-aware | out-degree | Dijkstra |
|---|---:|---:|---:|---:|
| jupyterlab → typing-extensions | 16 | **16** | **2** | 16 |
| flask → markupsafe | 7 | **7** | **2** | 7 |
| dvc → multidict | 100 | **100** | 96 | 100 |
| langchain → anyio | 21 | 21 | 21 | 21 |

The BFS-depth heuristic — the sophisticated one — **prunes nothing.** It expands exactly
as many nodes as Dijkstra on every pair. The trivial constant heuristic prunes 8× on two
of them.

**And the depth heuristic was originally wrong.** Its first version returned
`|level(goal) − level(n)|` with a confident admissibility proof attached in the docstring.
The proof only covers the case where the goal is *deeper*. When the goal is shallower —
`level(n)=5`, `level(goal)=1` — the absolute difference claims 4 hops remain while a
single edge may cost 1. That is inadmissible, and an inadmissible heuristic does not error:
**A* silently stops being optimal.**

`verify_admissible()` caught it against Dijkstra ground truth. Not a reviewer, not a test I
thought to write afterwards — the checker that exists precisely because admissibility
arguments written in docstrings are claims, and claims about search correctness are cheap
to check and expensive to get wrong.

---

## Real data, and a real cycle

The graph is fetched live from the **PyPI JSON API** — public, no credentials — across 31
seed packages: **411 packages, 989 edges, depths 0–9, closures of 1 to 109.**

An early root set of ordinary libraries topped out at **depth 4**. That is a genuine
property of modern pip packages once optional extras are excluded, and it was useless as a
sweep axis; the roots were widened to orchestration and ML stacks to get real depth.

Tarjan on the live graph found a genuine circular dependency in a major package family:

```
apache-airflow → apache-airflow-core → apache-airflow-providers-common-compat
  → apache-airflow-providers-common-io → apache-airflow-providers-common-sql
  → apache-airflow-providers-smtp → apache-airflow-providers-standard
  → apache-airflow-task-sdk → apache-airflow
```

No total order exists for that closure. The planner reports it rather than producing a
plausible-looking wrong answer.

---

## What is in it

| path | lines | what |
|---|---:|---|
| `graph/` | ~1,000 | Kahn, iterative Tarjan SCC, Dijkstra, A*, DAG longest-path, live PyPI builder, planner + grader |
| `rag/` | ~700 | two-kind corpus, MiniLM + hashed-TF-IDF embedders, pgvector + numpy stores, hybrid reranker |
| `agent/` | ~640 | context-measuring LLM adapter, tools bound to the algorithms, budgeted memory, the loop |
| `eval/` | ~560 | the depth sweep, the mechanism attribution, the report |
| `api/` | ~740 | FastAPI + CLI + the comparison UI |
| `tests/` | ~430 | 51 tests |
| `infra/` + CI + Makefile | ~250 | Dockerfile, compose with pgvector, Fly, GitHub Actions |

Every algorithm is **iterative**, not recursive: real dependency chains reach depths that
overflow CPython's 1,000-frame limit, and a resolver that crashes on a deep graph fails
exactly when the problem is hardest. A test drives a 4,000-node chain through all of them.

## Run it

```bash
make setup
make graph        # fetch 411 real packages from PyPI
make test         # 51 tests
make serve        # http://localhost:8300
```

The UI takes a package, resolves it exactly, then runs the agent on the same task and
shows both lists side by side — omissions in red, with the prompt-token count against the
context limit.

```bash
make sweep        # the depth sweep
make explain      # attribute the failures to a mechanism
make report       # consolidate into outputs/results.json
make docker       # with a real pgvector backend
python3 -m api.cli plan flask
python3 -m api.cli path jupyterlab typing-extensions
```

No API key is needed for anything: flan-t5-base and MiniLM both run locally on CPU.
Set `ANTHROPIC_API_KEY` to swap in a real API model.

## Known limits

- **One model.** The format effect is measured on flan-t5-base. A larger instruction-tuned
  model would very likely be robust to it — which is exactly why the finding is about
  *tool output design*, not about language models being unreliable.
- **Depth and closure size are confounded** in this graph; deeper packages also have bigger
  closures. The correlation table separates them as far as observational data allows, and
  no further.
- **Small samples at the deep end:** only 2 packages exist at depth 9 and 3 at depth 7.
  Those rows are indicative, not precise, and are reported with their `n`.
- **The agent has a tool that returns the exact answer.** This measures tool *use*, not
  reasoning. That is deliberate — it is what makes a failure interesting, because the
  answer was always available.
- **pgvector is implemented and schema-complete but the measured results use the numpy
  store.** At 822 documents an exact scan beats any approximate index, and pretending
  otherwise would be theatre.

See `DECISIONS.md` for why each algorithm was chosen and what was rejected.
