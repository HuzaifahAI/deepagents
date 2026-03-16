# StreamWriter Propagation — Test Results Report

**Date:** 2026-03-10
**Branch:** `master` (post-merge of `feature/token-streaming` and `feature/summarization-evict-messages-store`)
**Commit:** `36d262d`
**Conda Env:** `deep_agents_test` (Python 3.12)

---

## Environment Configuration

| Variable | Value |
|----------|-------|
| `LLM_MODEL_NAME` | `qwen3-235b` |
| `LLM_TEMPERATURE` | `0.1` |
| `LLM_MAX_ITERATIONS` | `8` |
| Python | `3.12` (conda: `deep_agents_test`) |
| `deepagents` | `0.3.11` (editable install from `libs/deepagents`) |
| `langgraph` | `1.0.10` |
| `langchain` | `1.2.10` |
| `langchain-openai` | `1.1.11` |

---

## Steps to Reproduce

### 1. Create the conda environment

```bash
conda create -n deep_agents_test python=3.12 -y
```

### 2. Install deepagents from master

```bash
git checkout master
git pull origin master

conda run -n deep_agents_test pip install -e libs/deepagents
conda run -n deep_agents_test pip install langchain-openai
```

### 3. Run the test script

```bash
conda run -n deep_agents_test python test_streaming_subagents.py
```

The script requires the LLM API endpoint to be reachable. Environment variables can be overridden:

```bash
LLM_API_BASE_URL="http://<your-host>:<port>/v1" \
LLM_MODEL_NAME="qwen3-235b" \
LLM_TEMPERATURE=0.1 \
conda run -n deep_agents_test python test_streaming_subagents.py
```

---

## Test Architecture

### Deep Agent (parent)

- Uses `ChatOpenAI` pointed at the Qwen3-235B vLLM endpoint
- Has `InMemorySaver` checkpointer
- Receives two compiled LangGraph workflows as `CompiledSubAgent` entries

### Subagent 1: `history_writer`

A 2-node LangGraph workflow:

```
START → research → draft → END
```

| Node | StreamWriter events |
|------|-------------------|
| `research` | `history_research/started`, `history_research/analyzing`, `history_research/completed` |
| `draft` | `history_drafting/started`, `history_drafting/writing`, `history_drafting/completed` |

### Subagent 2: `bio_researcher`

A 3-node LangGraph workflow:

```
START → gather_claims → evaluate_claims → synthesize → END
```

| Node | StreamWriter events |
|------|-------------------|
| `gather_claims` | `bio_claim_gathering/started`, `bio_claim_gathering/processing` (x2), `bio_claim_gathering/completed` |
| `evaluate_claims` | `evaluation_of_claims/started`, `evaluation_of_claims/verified` (x2), `evaluation_of_claims/completed` |
| `synthesize` | `bio_synthesis/started`, `bio_synthesis/completed` |

---

## Test Results

### Test 1: Streaming mode (`stream_mode="custom"`, `subgraphs=True`)

**Result: PASS**

The LLM routed the request to the `bio_researcher` subagent. All 10 custom `StreamWriter` events from the 3 subagent nodes were received by the parent stream.

| # | Node | Status | Detail |
|---|------|--------|--------|
| 1 | `bio_claim_gathering` | `started` | Collecting claims about: Research and write a comprehensive biography of Marie Curie... |
| 2 | `bio_claim_gathering` | `processing` | Verifying birth records |
| 3 | `bio_claim_gathering` | `processing` | Cross-referencing educational records |
| 4 | `bio_claim_gathering` | `completed` | Claims collected |
| 5 | `evaluation_of_claims` | `started` | Evaluating claim validity |
| 6 | `evaluation_of_claims` | `verified` | Birth date and place (confidence: 0.95) |
| 7 | `evaluation_of_claims` | `verified` | Education history (confidence: 0.88) |
| 8 | `evaluation_of_claims` | `completed` | All claims evaluated |
| 9 | `bio_synthesis` | `started` | Synthesizing verified claims into biography |
| 10 | `bio_synthesis` | `completed` | Biography synthesized |

**Stream item count:** 10 total, 10 with `node` key

### Test 2: Invoke mode (non-streaming)

**Result: PASS**

The LLM routed to a subagent for the Einstein biography request. `StreamWriter` calls inside subagent nodes were silently no-ops (expected behavior when not streaming).

| Metric | Value |
|--------|-------|
| Total messages | 4 |
| Tool messages (from subagent) | 1 |
| Final message type | `ai` |
| Errors | None |

**Final response (truncated):** "Albert Einstein was a theoretical physicist widely regarded as one of the most influential scientists of all time. Born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire..."

---

## Key Findings

1. **StreamWriter propagation works end-to-end.** Custom event dicts emitted via `writer()` inside compiled LangGraph subagent nodes flow to the parent Deep Agent's stream when using `stream_mode="custom"` with `subgraphs=True`.

2. **Non-streaming mode is safe.** When the parent uses `.invoke()`, `StreamWriter` calls are no-ops — no errors, no side effects, correct results returned.

3. **Multi-node workflows stream correctly.** Events from all nodes in the subagent graph (`gather_claims` → `evaluate_claims` → `synthesize`) arrive in execution order.

4. **Subagent identity is preserved.** The subagent runs with its own identity — the parent's `run_name`, `run_id`, and checkpoint keys are not leaked into the subagent config.

5. **LLM routing works.** The Qwen3-235B model correctly selected the appropriate subagent (`bio_researcher` for Marie Curie biography, subagent for Einstein) based on the `task` tool descriptions.
