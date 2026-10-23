"""Before/after latency report from logs/conversation_metadata.jsonl (no network, no extra packages).

  python latency_report.py            # every successful turn, grouped by pipeline version
  python latency_report.py --last 20  # only the 20 most recent turns

Turns recorded by the old, non-streaming pipeline (no time_to_first_audio_seconds) are reported
separately from the streaming pipeline, and the streaming turns are grouped by STT engine and models,
so a change of settings never gets averaged together with the old results.
"""
import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path

LOG = Path(__file__).resolve().parent / "logs" / "conversation_metadata.jsonl"


def load(last=None):
    rows = []
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    rows = [r for r in rows if r.get("status") == "success"]
    return rows[-last:] if last else rows


def group_key(r):
    if not (r.get("time_to_first_audio_seconds") or 0) > 0:
        return "OLD pipeline (waits for the whole answer + audio)"
    stt = (r.get("stt_backend") or "?").split(" (")[0]
    llm = (r.get("llm_model") or "?").split("/")[-1]
    return f"STREAMING  stt={stt}  llm={llm}  tts={r.get('tts_model') or '?'}"


def pick(r, *keys):
    for k in keys:
        v = r.get(k)
        if v:
            return float(v)
    return 0.0


def pctl(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))] if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, help="only the most recent N successful turns")
    args = ap.parse_args()
    rows = load(args.last)
    if not rows:
        print("No successful turns in", LOG)
        return
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)

    cols = [
        ("STT", lambda r: pick(r, "transcription_seconds")),
        ("LLM", lambda r: pick(r, "llm_first_sentence_seconds", "llm_seconds")),
        ("TTS", lambda r: pick(r, "tts_first_audio_seconds", "tts_generation_seconds")),
        ("FIRST AUDIO", lambda r: pick(r, "response_latency_seconds")),
    ]
    summary = {}
    for name, rs in groups.items():
        print(f"\n{name}   ({len(rs)} turns)")
        print(f"  {'':<14}{'mean':>8}{'median':>8}{'p90':>8}")
        for label, fn in cols:
            v = [fn(r) for r in rs]
            print(f"  {label:<14}{statistics.mean(v):8.2f}{statistics.median(v):8.2f}{pctl(v, 0.9):8.2f}")
        heard = [pick(r, "client_heard_after_seconds") for r in rs if r.get("client_heard_after_seconds")]
        if heard:
            print(f"  {'heard (browser)':<14}{statistics.mean(heard):8.2f}{statistics.median(heard):8.2f}{pctl(heard, 0.9):8.2f}")
        summary[name] = statistics.median(cols[3][1](r) for r in rs)

    old = [v for k, v in summary.items() if k.startswith("OLD")]
    new = [(k, v) for k, v in summary.items() if not k.startswith("OLD")]
    if old and new:
        best = min(new, key=lambda kv: kv[1])
        print(f"\nMedian first audio: OLD {old[0]:.2f} s -> best streaming setup {best[1]:.2f} s "
              f"({(old[0] - best[1]) / old[0] * 100:.0f}% faster)   [{best[0].split('  ', 1)[1].strip()}]")
    if len(new) > 1:
        print("Streaming setups ranked by median first audio:")
        for k, v in sorted(new, key=lambda kv: kv[1]):
            print(f"  {v:5.2f} s  {k.split('  ', 1)[1].strip()}")


if __name__ == "__main__":
    main()
