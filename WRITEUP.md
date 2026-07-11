# Intel Arc Pro B70 + vLLM XPU: Qwen3.6-27B INT4 at 32 tok/s decode, ~1.6-1.8k tok/s prefill (single GPU)

The community consensus on the Arc Pro B70 is that vLLM is the painful, slow path
and llama.cpp SYCL is the fast one. The published numbers back that up — ~13 tok/s
for vLLM vs ~22 tok/s for best-tuned llama.cpp SYCL on a 27B Q4. After getting
vLLM XPU working with Intel's own AutoRound INT4 quant inside Intel's
llm-scaler container, I'm measuring numbers well above both, so I'm publishing
the config and method.

**Headline: 32.2 tok/s decode at short context, 27.0 tok/s at 22.5k context,
with prefill throughput of ~1,550-1,780 tok/s throughout — on one B70, serving
a 27B model with a 32k context window.**

## Setup

| Component | Version |
|---|---|
| GPU | Intel Arc Pro B70 (Battlemage G31, 8086:e223) |
| CPU / RAM | AMD Ryzen 7 9700X / 32 GB DDR5 |
| Kernel | 7.0.0-27-generic, `xe` driver |
| Container | `intel/llm-scaler-vllm:0.14.0-b8.3.2` |
| vLLM | 0.14.1.dev0+gb17039bcc.d20260626.xpu |
| torch | 2.10.0+xpu |
| Model | `Intel/Qwen3.6-27B-int4-AutoRound` |

Exact serve command:

```
vllm serve Intel/Qwen3.6-27B-int4-AutoRound --dtype=float16 \
  --allow-deprecated-quantization --max-model-len 32768 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

The two decisions that made the difference, after several failed attempts with
other quants (AWQ community quants, FP8):

1. **Intel's own AutoRound INT4 quant** (`Intel/Qwen3.6-27B-int4-AutoRound`) —
   quant format matched to the XPU kernels, instead of NVIDIA-first community quants.
2. **Intel's llm-scaler-vllm container** instead of building vLLM XPU from source.

## Method

- Streaming requests via the OpenAI-compatible API (through llama-swap),
  `temperature=0`, 300 max output tokens, 3 runs per context length, medians reported.
- **Prefix caching defeated deliberately:** every run's prompt begins with a
  unique random line, so vLLM's APC can never reuse KV blocks across runs.
  Without this, repeat runs report near-zero TTFT and the prefill numbers are fiction.
- Token counts from the server's `usage` field, not client-side estimates.
- TTFT is end-to-end from request start (includes HTTP overhead and chat
  template) — the latency a user actually experiences.
- Single user, one request at a time. No concurrency/batching results here.

Benchmark script: single-file Python, stdlib only. [benchmark_longcontext.py](https://github.com/renewr2014/b70-benchmarks/blob/main/benchmark_longcontext.py)

## Results

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 488 | 0.30s | 1,643 | 32.2 |
| 1,874 | 1.05s | 1,782 | 31.6 |
| 7,493 | 4.48s | 1,673 | 30.1 |
| 14,939 | 9.31s | 1,605 | 28.5 |
| 22,460 | 14.42s | 1,558 | 27.0 |

Run-to-run variance was under ±1% at every length. Decode degrades only 16%
from 0.5k → 22.5k context. GPU peaked at 75°C package / 82°C VRAM.

### Memory and power

Captured 2026-07-11 during a re-run that reproduced the table above within
±0.5% (xpu-smi sampling at 1 s inside the serving container, plus vLLM's own
memory-profiling log; raw capture in `results/`):

- **Model weights on GPU: 17.89 GB**; reserved footprint after load: 21.57 GB.
- vLLM then preallocates the rest of the 32 GB card as its KV-cache pool, so
  monitoring tools report ~32.0 GB / 98% "used" while serving — that's the
  pool, **not the minimum needed to run the model**. The engine reports KV
  capacity for 34,304 full-attention tokens; the hybrid linear-attention
  layers keep constant-size state instead of growing KV with context.
- **Power: ~230 W sustained through both prefill and decode, 257 W peak,
  ~54 W idle with weights resident.** 230 W is exactly the reference
  design's rated TBP — the card is fully fed, holding 2800 MHz with no
  thermal or power throttling. At 22.5k context that works out to
  27 tok/s at ~230 W ≈ 0.12 tok/s per watt for long-context decode.

Cross-checks after the run:

- **Server-side logs agree.** vLLM's engine logs during the run show 10-second
  prompt-throughput averages that reconstruct each context bucket's token count
  exactly, clean decode windows at 31.2-31.3 tok/s matching the client-side
  measurement, and `Prefix cache hit rate: 0.0%` throughout — independent
  confirmation the anti-caching measure worked.
- **Output is coherent, not fast garbage.** A token counter can't tell 32 tok/s
  of good text from 32 tok/s of repetition loops. Spot-checked completions at
  long context: the model correctly summarizes the document and even calls out
  its repetitive filler structure.
- **Why long context holds up:** Qwen3.6-27B's config shows hybrid linear
  attention (Gated Delta Net `linear_attn` layers), so the flat decode curve is
  partly architecture, not purely the serving stack. It's also likely why this
  model family was reported as not running on mainline vLLM XPU at all —
  Intel's llm-scaler container carries the support that makes this possible.

## How this compares to published numbers

Careful framing: these are **not same-model, same-machine head-to-heads** — they
are my numbers next to the closest published ones. Model versions, quants, and
systems differ.

- Best-tuned llama.cpp SYCL, Qwen3.5-27B Q4, single B70: **22.5 tok/s** decode
  ([TeksEdge](https://x.com/TeksEdge/status/2044407998462738669))
- vLLM, Qwen3.5-27B Q4, single B70: **13.4 tok/s** decode (same source)
- llama.cpp SYCL prefill on a 14B Q4: **521-766 tok/s** (pp512-pp2048, noted as
  understated ~50% due to a build issue at the time)
  ([PMZFX engine comparison](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/engine-comparison.md))

Even reading the llama.cpp numbers charitably, this config decodes ~40% faster
than the best published SYCL result for a same-size model, and prefills ~2× faster
than llama.cpp numbers measured on a *smaller* model. If someone wants to run the
same model in llama.cpp SYCL on a B70 for a true head-to-head, I'd genuinely like
to see it — my benchmark script works against any OpenAI-compatible endpoint.

## Caveats

- `--enforce-eager` is on (required in my config at the time); there may be
  headroom without it if graph mode works for your setup.
- Filler text is a repeated paragraph (unique prefix per run). Diverse text
  could tokenize/attend slightly differently.
- Single-user only. vLLM's batching advantage isn't even being used here —
  concurrent-request numbers would only widen the gap.
- Qwen3.6-27B vs the published Qwen3.5-27B numbers: adjacent model generations,
  not identical architectures.

## Why publish

Every B70 guide I found either skips vLLM, calls it a cautionary tale, or
benchmarks it with a mismatched quant. The takeaway isn't "llama.cpp is slow" —
it's that **vLLM XPU's reputation on this card comes from configs that fight the
hardware**. Vendor-matched quant + vendor container, and it's the fastest
single-GPU path I've measured, with a 32k context window and OpenAI-compatible
serving for free.

---

*Script, raw results, and environment details: [github.com/renewr2014/b70-benchmarks](https://github.com/renewr2014/b70-benchmarks). I write about
running serious local AI on Intel Arc hardware at [renewr.substack.com](https://renewr.substack.com) — follow along
for the llama.cpp SYCL head-to-head and a full "27B agent on one Arc GPU"
build guide. If you want help setting up private, local AI for your home or
business, DM me on X: [@renewr20](https://x.com/renewr20).*
