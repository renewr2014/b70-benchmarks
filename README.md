# Intel Arc Pro B70 LLM Benchmarks

Benchmark results and tooling for LLM inference on the Intel Arc Pro B70
(Battlemage), focused on the vLLM XPU path with Intel's AutoRound INT4 quants.

Headline result (2026-07-10): **Qwen3.6-27B at 32 tok/s decode / ~1.6-1.8k tok/s
prefill on a single B70**, degrading only 16% in decode at 22.5k context.
Full writeup with method, environment, and caveats: [WRITEUP.md](WRITEUP.md).

## Contents

- `WRITEUP.md` — the full writeup: method, environment, results, caveats
- `benchmark_longcontext.py` — standalone benchmark script, stdlib only, works
  against any OpenAI-compatible endpoint
- `results/` — raw benchmark outputs with full environment headers
- `assets/` — headline chart + generating script

## Running the benchmark

```
python3 benchmark_longcontext.py MODEL_NAME --url http://127.0.0.1:9090
```

The script defeats server-side prefix caching (unique random prompt prefix per
run), takes token counts from the server's `usage` field, and reports median
TTFT / prefill / decode across runs. See the script docstring for the
methodology rationale.

## Planned follow-ups

- Same-model llama.cpp SYCL head-to-head (Qwen3.6-27B GGUF Q4_K_M, same
  script, same machine)
- Multi-model B70 roundup

Longer writeup and future posts:
[renewr.substack.com](https://renewr.substack.com) · questions/results
welcome via issues or [@renewr20](https://x.com/renewr20)
