"""Benchmark speech-to-text settings on THIS computer (free, offline; only --groq uses the network).

It times the same recording under your previous settings and the new ones, so you get a real
before/after for the biggest part of the latency. Check the transcripts too: a faster model is
only better if it still understands your Hindi and English.

Examples (PowerShell, from the voice_bot folder, venv active):
  python bench_stt.py --record 5                 # records an English then a Hindi clip, then benchmarks
  python bench_stt.py audio\\bench_en.wav audio\\bench_hi.wav
  python bench_stt.py --models base,tiny,small --threads 4,6,8
  python bench_stt.py --cuda --groq              # also try an NVIDIA GPU and Groq's free-tier Whisper
"""
import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import core

AUDIO_DIR = core.AUDIO_DIR


def record_clip(path, seconds, prompt):
    import scipy.io.wavfile as wavfile
    import sounddevice as sd
    input(f"\n{prompt}\nPress ENTER, then speak for {seconds} seconds... ")
    audio = sd.rec(int(seconds * core.SAMPLE_RATE), samplerate=core.SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    wavfile.write(str(path), core.SAMPLE_RATE, audio)
    print(f"saved {path}")


_models = {}


def norm(text):
    """Compare transcripts ignoring case and punctuation (incl. the Hindi danda)."""
    return re.sub(r"\s+", " ", re.sub(r"[.,!?;:।\"'()\-]", "", text.lower())).strip()


def load(model_name, device, compute, threads):
    key = (model_name, device, compute, threads)
    if key not in _models:
        from faster_whisper import WhisperModel
        print(f"  loading {model_name} ({device}, {compute}, threads={threads}) ...", flush=True)
        _models[key] = WhisperModel(model_name, device=device, compute_type=compute, cpu_threads=threads)
    return _models[key]


def run_local(cfg, path):
    model = load(cfg["model"], cfg["device"], cfg["compute"], cfg["threads"])
    kw = dict(beam_size=1, language=None, condition_on_previous_text=False, without_timestamps=True)
    if cfg["temp0"]:
        kw["temperature"] = 0.0
    if cfg["vad"]:
        kw.update(vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    t0 = time.perf_counter()
    segments, info = model.transcribe(str(path), **kw)
    text = " ".join(s.text.strip() for s in segments).strip()
    return time.perf_counter() - t0, text, info.language


def run_groq(cfg, path):
    data = Path(path).read_bytes()
    t0 = time.perf_counter()
    text, lang = core._transcribe_groq(data, Path(path).suffix or ".wav")
    return time.perf_counter() - t0, text, lang


def build_configs(args):
    base_model = core.WHISPER_MODEL
    models = [base_model] + [m for m in args.models.split(",") if m and m != base_model]
    cfgs = [
        dict(name=f"BEFORE: {base_model}, 4 threads, temperature fallback", model=base_model, device="cpu",
             compute="int8", threads=0, temp0=False, vad=True, kind="local", before=True),
        dict(name=f"AFTER:  {base_model}, {core.WHISPER_THREADS} threads, temperature 0", model=base_model, device="cpu",
             compute="int8", threads=core.WHISPER_THREADS, temp0=True, vad=True, kind="local"),
        dict(name=f"AFTER:  {base_model}, {core.WHISPER_THREADS} threads, temperature 0, VAD off", model=base_model,
             device="cpu", compute="int8", threads=core.WHISPER_THREADS, temp0=True, vad=False, kind="local"),
    ]
    for t in [int(x) for x in args.threads.split(",") if x]:
        if t != core.WHISPER_THREADS:
            cfgs.append(dict(name=f"AFTER:  {base_model}, {t} threads", model=base_model, device="cpu",
                             compute="int8", threads=t, temp0=True, vad=True, kind="local"))
    for m in models[1:]:
        cfgs.append(dict(name=f"AFTER:  {m}, {core.WHISPER_THREADS} threads", model=m, device="cpu",
                         compute="int8", threads=core.WHISPER_THREADS, temp0=True, vad=True, kind="local"))
    if args.cuda:
        if core._cuda_available():
            for m in models:
                cfgs.append(dict(name=f"GPU:    {m}, float16", model=m, device="cuda", compute="float16",
                                 threads=core.WHISPER_THREADS, temp0=True, vad=True, kind="local"))
        else:
            print("No NVIDIA GPU / CUDA libraries detected by CTranslate2; skipping --cuda.")
    if args.groq:
        if core.GROQ_API_KEY:
            cfgs.append(dict(name=f"CLOUD:  Groq {core.GROQ_STT_MODEL} (free tier, audio leaves this PC)", kind="groq"))
        else:
            print("GROQ_API_KEY missing; skipping --groq.")
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="wav/webm/mp3 recordings (default: audio/bench_en.wav and audio/bench_hi.wav)")
    ap.add_argument("--record", type=int, metavar="SECONDS", help="record an English and a Hindi clip first")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per file after one warm-up (default 3)")
    ap.add_argument("--models", default="tiny", help="extra models to compare, e.g. tiny,small (default: tiny)")
    ap.add_argument("--threads", default="", help="extra thread counts to try, e.g. 4,6,8")
    ap.add_argument("--cuda", action="store_true", help="also try an NVIDIA GPU")
    ap.add_argument("--groq", action="store_true", help="also time Groq's hosted Whisper (needs GROQ_API_KEY)")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    if args.record:
        AUDIO_DIR.mkdir(exist_ok=True)
        record_clip(AUDIO_DIR / "bench_en.wav", args.record, "Say an ENGLISH sentence, for example: What is machine learning?")
        record_clip(AUDIO_DIR / "bench_hi.wav", args.record, "Say a HINDI sentence, for example: मुझे पायथन के बारे में बताइए")
        files = [AUDIO_DIR / "bench_en.wav", AUDIO_DIR / "bench_hi.wav"]
    if not files:
        files = [p for p in (AUDIO_DIR / "bench_en.wav", AUDIO_DIR / "bench_hi.wav") if p.exists()]
    if not files:
        sys.exit("No recordings. Run:  python bench_stt.py --record 5   or pass file paths.")

    cfgs = build_configs(args)
    results = {}   # name -> {file: (median, text, lang)}
    for cfg in cfgs:
        print(f"\n== {cfg['name']}")
        results[cfg["name"]] = {}
        for f in files:
            try:
                fn = run_groq if cfg["kind"] == "groq" else run_local
                fn(cfg, f)                                   # warm-up, not counted
                times, text, lang = [], "", ""
                for _ in range(args.runs):
                    secs, text, lang = fn(cfg, f)
                    times.append(secs)
                results[cfg["name"]][f.name] = (statistics.median(times), text, lang)
                print(f"  {f.name:<16} {statistics.median(times):6.2f} s  [{lang}]  {text[:70]}")
            except Exception as exc:
                print(f"  {f.name:<16} FAILED: {type(exc).__name__}: {exc}")

    # ---- summary
    print("\n" + "=" * 100)
    ref_name = next(c["name"] for c in cfgs if c["name"].startswith("AFTER:") and "VAD off" not in c["name"]
                    and c["name"] in results and len(results[c["name"]]) == len(files))
    ref = {f: norm(v[1]) for f, v in results[ref_name].items()}
    print(f"{'configuration':<66}{'mean s':>8}  {'vs BEFORE':>10}  same text as '{core.WHISPER_MODEL}'?")
    before = None
    rows = []
    for cfg in cfgs:
        r = results[cfg["name"]]
        if len(r) != len(files):
            continue
        mean = statistics.mean(v[0] for v in r.values())
        same = all(norm(r[f][1]) == ref[f] for f in r)
        rows.append((cfg, mean, same))
        if cfg.get("before"):
            before = mean
    for cfg, mean, same in rows:
        delta = f"{(before - mean) / before * 100:+.0f}%" if before and not cfg.get("before") else ""
        print(f"{cfg['name']:<66}{mean:8.2f}  {delta:>10}  {'yes' if same else 'NO - check the transcript'}")
    local = [(c, m) for c, m, same in rows if c["kind"] == "local" and not c.get("before") and same]
    if local:
        best, mean = min(local, key=lambda x: x[1])
        print("\nFastest local configuration with the same transcripts:", best["name"].strip(), f"({mean:.2f} s)")
        print("Put this in .env (and listen to a Hindi test in the web UI before keeping it):")
        print(f"  WHISPER_MODEL={best['model']}")
        if best["device"] == "cuda":
            print("  WHISPER_DEVICE=cuda")
        else:
            print(f"  WHISPER_DEVICE=cpu\n  WHISPER_THREADS={best['threads']}")
        print(f"  WHISPER_VAD={'1' if best['vad'] else '0'}")
    faster_but_different = [c["name"].strip() for c, m, same in rows if not same and c["kind"] == "local" and before and m < before]
    if faster_but_different:
        print("\nFaster but the text changed (usually lower accuracy, especially for Hindi):")
        for n in faster_but_different:
            print("  -", n)
    print("\nRemember: the bot's STT time adds up with ~0.45 s LLM and ~0.4 s TTS before you hear the first word.")


if __name__ == "__main__":
    main()
