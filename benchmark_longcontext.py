#!/usr/bin/env python3
"""Long-context LLM benchmark for any OpenAI-compatible endpoint. Stdlib only.

Measures, per context length: TTFT (end-to-end time to first streamed token),
prefill throughput (prompt_tokens / TTFT), and decode rate
(completion_tokens / time after first token).

Credibility notes, deliberate choices:
- Every run's prompt starts with a unique random line, so server-side prefix
  caching (e.g. vLLM APC) can never reuse KV blocks across runs — block hashes
  include all preceding context, so a unique first line invalidates everything
  after it. Without this, run 2+ of an identical prompt reports a near-zero
  TTFT that says nothing about real prefill speed.
- Token counts come from the server's usage field, not client-side estimates.
- TTFT is end-to-end from request start (includes HTTP + chat-template +
  queueing) — the honest number a user actually experiences.

Usage:
    python3 benchmark_longcontext.py MODEL_NAME [--url http://host:port] \
        [--lengths 512 2048 8192 16384 24576] [--runs 3]
"""

import argparse
import datetime
import json
import platform
import random
import string
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_URL = "http://127.0.0.1:9090"
DEFAULT_LENGTHS = [512, 2048, 8192, 16384, 24576]
MAX_DECODE_TOKENS = 300

# ~60 words ≈ 75-85 tokens on Qwen-family tokenizers. Repeated to approximate
# each target length; actual prompt_tokens is reported from the API, targets
# are just buckets.
_PARAGRAPH = (
    "The storage subsystem exposes a block interface to the scheduler, which "
    "batches incoming write requests into fixed-size segments before they are "
    "committed to the journal. Each segment carries a checksum and a sequence "
    "number so that recovery after an unclean shutdown can replay the journal "
    "deterministically. Compaction runs in the background and merges adjacent "
    "segments when their live-data ratio falls below the configured threshold. "
)


def build_prompt(target_tokens: int, run_seed: str) -> str:
    repeats = max(1, target_tokens // 80)
    doc = _PARAGRAPH * repeats
    return (
        f"Document ID: {run_seed}\n\n{doc}\n\n"
        "Summarize the document above in 3-4 sentences, then note anything "
        "unusual about its structure."
    )


@dataclass
class LongRunResult:
    prompt_tokens: int
    completion_tokens: int
    ttft: float
    prefill_tps: float
    decode_tps: float
    total_time: float
    error: Optional[str] = None


def do_request(base_url: str, model: str, prompt_text: str, timeout: int = 600) -> LongRunResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": MAX_DECODE_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        t_end = time.perf_counter()
        if t_first is None:
            return LongRunResult(prompt_tokens, 0, 0, 0, 0, t_end - t_start,
                                 error="no content streamed")
        ttft = t_first - t_start
        decode_time = t_end - t_first
        return LongRunResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft=ttft,
            prefill_tps=prompt_tokens / ttft if ttft > 0 else 0.0,
            decode_tps=completion_tokens / decode_time if decode_time > 0 else 0.0,
            total_time=t_end - t_start,
        )
    except Exception as e:
        return LongRunResult(0, 0, 0, 0, 0, time.perf_counter() - t_start, error=str(e))


def warmup(base_url: str, model: str, timeout: int = 600) -> None:
    """Short request to trigger model load; excluded from measurements."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "stream": False,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except Exception as e:
        print(f"  warmup error: {e}")


class TempMonitor:
    """Samples GPU hwmon temperature sensors once per second, if present.
    Silently records nothing on systems without matching sysfs paths."""

    def __init__(self, hwmon_glob: str = "/sys/class/drm/card*/device/hwmon/hwmon*"):
        self._paths = {}
        for hwmon in sorted(Path("/").glob(hwmon_glob.lstrip("/"))):
            for f in sorted(hwmon.glob("temp*_input")):
                label_file = f.with_name(f.name.replace("_input", "_label"))
                label = label_file.read_text().strip() if label_file.exists() else f.name
                self._paths[f"{hwmon.name}/{label}"] = f
            if self._paths:
                break
        self.readings: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self._paths:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(1.0):
            entry = {}
            for name, path in self._paths.items():
                try:
                    entry[name] = int(path.read_text().strip()) / 1000.0
                except Exception:
                    pass
            if entry:
                self.readings.append(entry)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        out: dict = {}
        for r in self.readings:
            for k, v in r.items():
                out.setdefault(k, []).append(v)
        return {k: {"min": min(v), "avg": sum(v) / len(v), "max": max(v)}
                for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description="Long-context benchmark for OpenAI-compatible endpoints")
    parser.add_argument("model")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--lengths", type=int, nargs="*", default=DEFAULT_LENGTHS)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print("=" * 72)
    print(f"LONG-CONTEXT BENCHMARK — {args.model} @ {args.url}")
    print("=" * 72)
    print(f"Date: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Host: {platform.platform()}")
    print("Record your GPU, driver, serving stack, and exact serve flags alongside")
    print("these results — numbers without an environment are not reproducible.")
    print("=" * 72)

    print("\nLoading model (warmup)...", flush=True)
    t_load = time.perf_counter()
    warmup(args.url, args.model)
    print(f"Model ready in {time.perf_counter() - t_load:.1f}s", flush=True)

    monitor = TempMonitor()
    monitor.start()
    results: dict[int, list[LongRunResult]] = {}

    for target in args.lengths:
        results[target] = []
        for run in range(args.runs):
            seed = "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
            prompt_text = build_prompt(target, seed)
            print(f"[~{target} tok] run {run + 1}/{args.runs}... ", end="", flush=True)
            r = do_request(args.url, args.model, prompt_text)
            results[target].append(r)
            if r.error:
                print(f"ERROR: {r.error}")
            else:
                print(f"prompt={r.prompt_tokens}  TTFT={r.ttft:.2f}s  "
                      f"prefill={r.prefill_tps:.0f} tok/s  decode={r.decode_tps:.1f} tok/s")

    monitor.stop()

    print("\n" + "=" * 72)
    print("RESULTS (median of runs)")
    print("=" * 72)
    print(f"{'Prompt tok':>10}  {'TTFT':>7}  {'Prefill tok/s':>13}  {'Decode tok/s':>12}")
    print("-" * 50)
    for target, runs in results.items():
        ok = sorted([r for r in runs if not r.error], key=lambda r: r.ttft)
        if not ok:
            print(f"{target:>10}  ERROR: {runs[0].error}")
            continue
        med = ok[len(ok) // 2]
        print(f"{med.prompt_tokens:>10}  {med.ttft:>6.2f}s  {med.prefill_tps:>13.0f}  {med.decode_tps:>12.1f}")

    temps = monitor.summary()
    if temps:
        print("\nTemps during run (°C):")
        for k, v in temps.items():
            print(f"  {k:<24} min={v['min']:.0f} avg={v['avg']:.0f} max={v['max']:.0f}")


if __name__ == "__main__":
    main()
