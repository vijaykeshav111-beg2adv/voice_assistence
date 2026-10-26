"""Shared logic for the CLI (main.py) and the web server (server.py).

Latency design (see README "How the latency was reduced"):
  * STT   : Groq-hosted Whisper (fast) with automatic fallback to local faster-whisper.
  * LLM   : streamed, so we can start speaking before the answer is finished.
  * TTS   : the answer is cut into sentences; each sentence is synthesised in parallel and
            sent to the client as soon as it is ready (first audio after ~1 sentence, not
            after the whole answer).
  * Every stage is timed and written to logs/voicebot.log (LATENCY lines) and to
    logs/conversation_metadata.jsonl / .csv.
"""
import csv
import json
import logging
import os
import re
import threading
import time
import asyncio
import edge_tts
import uuid  # noqa: F401  (kept for callers that import it from core)
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------- configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")      # fastest model that supports Hindi
TTS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_22050_32")  # small = quick to transfer
TTS_CONCURRENCY = max(1, int(os.getenv("TTS_CONCURRENCY", "2")))        # parallel TTS requests per server

STT_BACKEND = os.getenv("STT_BACKEND", "groq").strip().lower()          # "groq" or "local"
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_STT_TIMEOUT = float(os.getenv("GROQ_STT_TIMEOUT", "8"))           # seconds before falling back to local
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")                      # local fallback only
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "") or None            # e.g. "hi" / "en"; empty = auto-detect
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto").strip().lower()    # auto | cpu | cuda (NVIDIA GPU)
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "").strip()              # empty = int8 on CPU, float16 on GPU
WHISPER_VAD = os.getenv("WHISPER_VAD", "1").strip() not in ("0", "false", "no")
# CPU threads. CTranslate2's default is 4, which leaves most modern laptops idle; roughly the number
# of physical cores (logical // 2) is usually fastest. Override with WHISPER_THREADS (0 = library default).
_auto_threads = max(4, min(8, (os.cpu_count() or 8) // 2))
try:
    WHISPER_THREADS = int(os.getenv("WHISPER_THREADS") or _auto_threads)
except ValueError:                      # e.g. a stray comment pasted after the "="
    WHISPER_THREADS = _auto_threads

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")         # gpt-oss only: low / medium / high

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "8"))                  # CLI: maximum recording length
MAX_WEB_RECORD_SECONDS = int(os.getenv("MAX_WEB_RECORD_SECONDS", "30"))

AUDIO_DIR = BASE_DIR / "audio"
LOG_DIR = BASE_DIR / "logs"
METADATA_JSONL = LOG_DIR / "conversation_metadata.jsonl"
METADATA_CSV = LOG_DIR / "conversation_metadata.csv"
AUDIO_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "voicebot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)
# The per-request HTTP lines from these libraries drowned out the useful lines.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "faster_whisper", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


SYSTEM_PROMPT = (
    "You are a concise, natural voice assistant for a doctor appointment "
    "booking application. Respond to the user's latest message in plain text. "
    "No tools or functions are available in this language-model request. "
    "Do not call, request, or emit tool or function calls. "
    "Do not claim to book, reschedule, cancel, or check database records. "
    "The application's separate backend handles appointment operations. "
    "Keep responses natural, helpful, and conversational."
)


# Order of the CSV columns. The CSV is regenerated from the JSONL on every write, so adding a
# column here never breaks older rows (they just show an empty cell).
CSV_FIELDS = [
    "turn_id", "source", "timestamp_start_local", "timestamp_end_local",
    "date", "time", "timezone", "language", "user_input", "bot_response",
    "stt_backend", "llm_model", "tts_model",
    "recording_seconds", "transcription_seconds",
    "llm_first_token_seconds", "llm_first_sentence_seconds", "llm_seconds",
    "tts_first_audio_seconds", "tts_generation_seconds",
    "time_to_first_audio_seconds", "server_total_seconds", "client_heard_after_seconds",
    "response_latency_seconds", "playback_seconds", "total_seconds",
    "sentences", "status", "error",
]
TIMING_FIELDS = [
    "recording_seconds", "transcription_seconds", "llm_first_token_seconds",
    "llm_first_sentence_seconds", "llm_seconds", "tts_first_audio_seconds",
    "tts_generation_seconds", "time_to_first_audio_seconds", "server_total_seconds",
    "client_heard_after_seconds", "playback_seconds",
]


# ---------------------------------------------------------------- lazy clients
_groq = _eleven = _whisper = None
_whisper_lock = threading.Lock()
_client_lock = threading.Lock()


def missing_keys():
    return [k for k, v in (("GROQ_API_KEY", GROQ_API_KEY),
                           ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)) if not v]


def require_keys():
    miss = missing_keys()
    if miss:
        raise RuntimeError(f"{', '.join(miss)} missing. Copy .env.example to .env and add your keys.")


def _http_client():
    """One pooled HTTP client per service. httpx drops idle connections after 5 s by default, which
    forces a fresh TCP+TLS handshake (often 200-500 ms) on most turns; keep them for 2 minutes."""
    import httpx
    return httpx.Client(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120.0),
    )


def groq_client():
    global _groq
    with _client_lock:
        if _groq is None:
            from groq import Groq
            try:
                _groq = Groq(api_key=GROQ_API_KEY, http_client=_http_client())
            except TypeError:
                _groq = Groq(api_key=GROQ_API_KEY)
        return _groq


def eleven_client():
    global _eleven
    with _client_lock:
        if _eleven is None:
            from elevenlabs.client import ElevenLabs
            try:
                _eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY, httpx_client=_http_client())
            except TypeError:
                _eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        return _eleven


_whisper_force_cpu = False


def _cuda_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _whisper_target():
    """(device, compute_type) for the local model."""
    use_gpu = (not _whisper_force_cpu) and (WHISPER_DEVICE == "cuda" or (WHISPER_DEVICE == "auto" and _cuda_available()))
    device = "cuda" if use_gpu else "cpu"
    return device, (WHISPER_COMPUTE or ("float16" if use_gpu else "int8"))


def whisper_model():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            device, compute = _whisper_target()
            logging.info("Loading Whisper '%s' on %s (%s, threads=%s)", WHISPER_MODEL, device, compute, WHISPER_THREADS)
            _whisper = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute, cpu_threads=WHISPER_THREADS)
        return _whisper


_last_warm = 0.0


def prewarm(force=False):
    """Open the HTTPS connections to Groq and ElevenLabs in the background so the first real request
    of a turn does not pay for DNS + TCP + TLS. Cheap GETs, no credits used. Safe to call often."""
    global _last_warm
    now = time.time()
    if missing_keys() or (not force and now - _last_warm < 20):
        return
    _last_warm = now

    def _go():
        try:
            groq_client().models.list()
        except Exception as exc:
            logging.debug("Groq warm-up skipped: %s", exc)
        try:
            eleven_client().voices.get(VOICE_ID)
        except Exception as exc:
            logging.debug("ElevenLabs warm-up skipped: %s", exc)

    threading.Thread(target=_go, name="prewarm", daemon=True).start()


def warm_local_whisper():
    """Load the local model and run one dummy inference so the first real turn is not slower.
    If the GPU was chosen but its CUDA libraries are missing, fall back to the CPU automatically."""
    global _whisper, _whisper_force_cpu
    import numpy as np
    for _ in range(2):
        try:
            model = whisper_model()
            with _whisper_lock:
                segs, _info = model.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"), beam_size=1, language="en")
                list(segs)
            device, compute = _whisper_target()
            print(f"Local Whisper ready: {WHISPER_MODEL} on {device} ({compute})")
            return
        except Exception as exc:
            if _whisper_target()[0] != "cuda":
                raise
            logging.warning("GPU Whisper failed (%s); using the CPU instead", exc)
            print("GPU Whisper unavailable, using the CPU instead.")
            _whisper, _whisper_force_cpu = None, True


# ---------------------------------------------------------------- speech to text
_LANG_NAMES = {"english": "en", "hindi": "hi"}


def _norm_lang(lang):
    lang = (lang or "").strip().lower()
    return _LANG_NAMES.get(lang, lang)


def _transcribe_groq(audio_bytes, suffix):
    kwargs = dict(
        file=(f"speech{suffix}", audio_bytes),
        model=GROQ_STT_MODEL,
        response_format="verbose_json",
        temperature=0.0,
    )
    if WHISPER_LANGUAGE:
        kwargs["language"] = WHISPER_LANGUAGE
    client = groq_client()
    try:   # no hidden retries/back-off: if Groq is not answering fast, fall back to local right away
        client = client.with_options(max_retries=0, timeout=GROQ_STT_TIMEOUT)
    except AttributeError:
        pass
    resp = client.audio.transcriptions.create(**kwargs)
    if isinstance(resp, dict):
        return (resp.get("text") or "").strip(), _norm_lang(resp.get("language"))
    return (getattr(resp, "text", "") or "").strip(), _norm_lang(getattr(resp, "language", ""))


def _transcribe_local(audio_bytes, suffix):
    import io
    model = whisper_model()
    kwargs = dict(
        beam_size=1, language=WHISPER_LANGUAGE,
        temperature=0.0,                 # default is a list of 6 temperatures: a "failed" decode is silently
                                         # retried up to 5 more times, which multiplies the decode time
        condition_on_previous_text=False, without_timestamps=True,
    )
    if WHISPER_VAD:
        kwargs.update(vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    with _whisper_lock:  # the model is not safe to call from several threads at once
        segments, info = model.transcribe(io.BytesIO(audio_bytes), **kwargs)
        text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language


def transcribe_audio(audio_bytes, suffix=".webm"):
    """Return (text, language, seconds, backend). Groq first, local Whisper as fallback.
    `seconds` includes any time wasted on a failed Groq attempt, because the user waited for it,
    and `backend` says so ("local:base (groq failed after 0.31s: NotFoundError)")."""
    started = time.perf_counter()
    note = ""
    if STT_BACKEND == "groq":
        try:
            text, lang = _transcribe_groq(audio_bytes, suffix)
            return text, lang, time.perf_counter() - started, f"groq:{GROQ_STT_MODEL}"
        except Exception as exc:
            note = f" (groq failed after {time.perf_counter() - started:.2f}s: {type(exc).__name__})"
            logging.exception("Groq transcription failed; falling back to local Whisper")
    text, lang = _transcribe_local(audio_bytes, suffix)
    return text, lang, time.perf_counter() - started, f"local:{WHISPER_MODEL}{note}"


# ---------------------------------------------------------------- text helpers
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BULLET = re.compile(r"^\s*(?:[-*•]|#{1,6})\s+", re.M)
_SYMBOLS = re.compile(r"[*_`#~|<>]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
_SPACES = re.compile(r"\s+")


def clean_for_speech(text):
    """Strip markdown/emoji so the TTS does not read out 'asterisk asterisk'."""
    text = _LINK.sub(r"\1", text)
    text = _BULLET.sub("", text)
    text = _SYMBOLS.sub("", text)
    text = _EMOJI.sub("", text)
    return _SPACES.sub(" ", text).strip()


class SentenceChunker:
    """Cuts a streamed answer into speakable pieces as early as is safe.

    The first piece is allowed to be short (and to end at a comma once it is long enough) so the
    first audio starts quickly; later pieces are longer for smoother prosody and fewer API calls.
    """
    END = ".!?।॥…"
    CLOSERS = "\"')]}”’»"
    CLAUSE = ",;:—–"
    FIRST_MIN, NEXT_MIN = 12, 40
    FIRST_CLAUSE_MIN = 24
    MAX_CHARS = 200

    def __init__(self):
        self.buf = ""
        self.emitted = 0

    def feed(self, text):
        self.buf += text
        out = []
        while True:
            cut = self._cut()
            if cut is None:
                break
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            if piece.strip():
                out.append(piece.strip())
                self.emitted += 1
        return out

    def flush(self):
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []

    def _cut(self):
        buf, n = self.buf, len(self.buf)
        minlen = self.FIRST_MIN if self.emitted == 0 else self.NEXT_MIN
        i = 0
        while i < n:
            ch = buf[i]
            j = None
            if ch == "\n":
                j = i + 1
            elif ch in self.END:
                j = i + 1
                while j < n and (buf[j] in self.END or buf[j] in self.CLOSERS):
                    j += 1
                # Only a boundary if followed by whitespace ("3.5" and "e.g.x" are not).
                # If the buffer ends right here we must wait for the next character.
                if not (j < n and buf[j].isspace()):
                    j = None
            if j is not None and len(buf[:j].strip()) >= minlen:
                return j
            i += 1 if j is None else max(1, j - i)
        # No sentence end yet: allow an early clause break for the very first piece, or break a
        # run-on sentence so the audio is never held back for long.
        if self.emitted == 0 and n >= self.FIRST_CLAUSE_MIN:
            for k in range(self.FIRST_CLAUSE_MIN - 1, n - 1):
                if buf[k] in self.CLAUSE and buf[k + 1].isspace():
                    return k + 1
        if n >= self.MAX_CHARS:
            k = max(buf.rfind(c, 0, self.MAX_CHARS) for c in self.CLAUSE)
            if k < 60:
                k = buf.rfind(" ", 0, self.MAX_CHARS)
            return k + 1 if k > 0 else self.MAX_CHARS
        return None


# ---------------------------------------------------------------- language model
def new_conversation():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _llm_extra():
    """Reasoning models (gpt-oss) 'think' before the first visible token. Low effort and hidden
    reasoning cut that delay a lot, and a spoken answer does not need deep reasoning."""
    if "gpt-oss" in GROQ_MODEL.lower():
        extra = {"include_reasoning": False}
        if LLM_REASONING_EFFORT:
            extra["reasoning_effort"] = LLM_REASONING_EFFORT
        return extra
    return {}


def _open_llm_stream(messages):
    kwargs = dict(model=GROQ_MODEL, messages=messages, temperature=0.5,
                  max_tokens=LLM_MAX_TOKENS, stream=True)
    extra = _llm_extra()
    try:
        return groq_client().chat.completions.create(**kwargs, **({"extra_body": extra} if extra else {}))
    except Exception as exc:
        if extra and getattr(exc, "status_code", None) == 400:
            logging.warning("Model rejected reasoning options (%s); retrying without them", exc)
            return groq_client().chat.completions.create(**kwargs)
        raise


def _delta_text(chunk):
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


# ---------------------------------------------------------------- text to speech
_tts_executor = None
_tts_executor_lock = threading.Lock()


def _tts_pool():
    global _tts_executor
    with _tts_executor_lock:
        if _tts_executor is None:
            _tts_executor = ThreadPoolExecutor(max_workers=TTS_CONCURRENCY, thread_name_prefix="tts")
        return _tts_executor


def _tts_clip(text):
    """Synthesise one piece of text. Returns {"audio", "first_byte", "seconds", "done_at"}."""
    last_exc = None
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            parts, first = [], None
            for part in eleven_client().text_to_speech.convert(
                    voice_id=VOICE_ID, text=text, model_id=TTS_MODEL, output_format=TTS_OUTPUT_FORMAT):
                if part and first is None:
                    first = time.perf_counter() - started
                parts.append(part)
            audio = b"".join(parts)
            if not audio:
                raise RuntimeError("ElevenLabs returned no audio")
            done = time.perf_counter()
            return {"audio": audio, "first_byte": first or 0.0, "seconds": done - started, "done_at": done}
        except Exception as exc:
            last_exc = exc
            logging.warning("TTS attempt %d failed: %s", attempt, exc)
            if attempt == 1:
                time.sleep(0.25)
    raise last_exc


def synthesize_speech(text):
    """Return (mp3_bytes, seconds). Used for one-off phrases such as the CLI greeting."""
    res = _tts_clip(text)
    return res["audio"], res["seconds"]


_EXIT_PHRASES = ["stop voice bot", "exit voice bot", "बंद करो", "बंद कर दो", "अलविदा"]


def is_exit_command(text):
    """Only treat short utterances as exit commands, so a long sentence that merely contains
    'बंद करो' does not shut the bot down."""
    t = text.lower().strip()
    return len(t.split()) <= 5 and any(p in t for p in _EXIT_PHRASES)


# ---------------------------------------------------------------- metadata + timing log
_meta_lock = threading.Lock()


def now_local_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def new_record(turn_id, source="voice"):
    """Create a record. Call this when the turn actually starts (after the user pressed record)."""
    now = datetime.now().astimezone()
    rec = {
        "turn_id": turn_id, "source": source,
        "timestamp_start_local": now.isoformat(timespec="milliseconds"),
        "timestamp_end_local": "",
        "date": now.date().isoformat(), "time": now.strftime("%H:%M:%S"), "timezone": now.strftime("%z"),
        "language": "", "user_input": "", "bot_response": "",
        "stt_backend": "", "llm_model": GROQ_MODEL, "tts_model": TTS_MODEL,
        "response_latency_seconds": 0.0, "total_seconds": 0.0, "sentences": 0,
        "status": "started", "error": "",
    }
    for k in TIMING_FIELDS:
        rec[k] = 0.0
    return rec


def finalize_latency(rec):
    """response_latency = time from the end of the user's speech until the FIRST audio of the reply
    was ready (the number the user actually feels). total = recording + that + playback span.
    Older records / failed turns without a first-audio time fall back to STT + LLM + TTS."""
    first_audio = rec.get("time_to_first_audio_seconds") or 0.0
    if first_audio > 0:
        latency = first_audio
    else:
        latency = ((rec.get("transcription_seconds") or 0) + (rec.get("llm_seconds") or 0)
                   + (rec.get("tts_generation_seconds") or 0))
    rec["response_latency_seconds"] = round(latency, 4)
    rec["total_seconds"] = round((rec.get("recording_seconds") or 0) + latency
                                 + (rec.get("playback_seconds") or 0), 4)


def log_timing(rec):
    """One greppable line per turn in logs/voicebot.log."""
    f = lambda k: rec.get(k) or 0.0  # noqa: E731
    logging.info(
        "LATENCY turn=%s src=%s status=%s lang=%s | stt=%.2fs llm_first_token=%.2fs "
        "llm_first_sentence=%.2fs llm_total=%.2fs tts_first_clip=%.2fs tts_total=%.2fs | "
        "FIRST_AUDIO=%.2fs server_total=%.2fs | recording=%.2fs audio=%sKB | sentences=%s | stt_backend=%s llm=%s tts=%s%s",
        str(rec.get("turn_id"))[:8], rec.get("source"), rec.get("status"), rec.get("language") or "-",
        f("transcription_seconds"), f("llm_first_token_seconds"), f("llm_first_sentence_seconds"),
        f("llm_seconds"), f("tts_first_audio_seconds"), f("tts_generation_seconds"),
        f("time_to_first_audio_seconds"), f("server_total_seconds"), f("recording_seconds"),
        rec.get("audio_kb", "-"), rec.get("sentences", 0), rec.get("stt_backend") or "-", rec.get("llm_model"), rec.get("tts_model"),
        f" | error={rec['error']}" if rec.get("error") else "",
    )


def finish_turn(rec):
    """Stamp the end time, round, compute totals, save to JSONL/CSV and write the LATENCY log line."""
    rec["timestamp_end_local"] = now_local_iso()
    for k in TIMING_FIELDS:
        rec[k] = round(float(rec.get(k) or 0.0), 4)
    finalize_latency(rec)
    persist_metadata(rec)
    log_timing(rec)


def report_playback(turn_id, playback_seconds, heard_after_seconds=None):
    """Called once the reply has finished playing (browser or CLI). Updates the record and logs it."""
    fields = {"playback_seconds": round(max(0.0, min(float(playback_seconds), 3600.0)), 4)}
    if heard_after_seconds is not None:
        fields["client_heard_after_seconds"] = round(max(0.0, min(float(heard_after_seconds), 600.0)), 4)
    rec = update_metadata(turn_id, fields)
    if rec is not None:
        logging.info(
            "PLAYBACK turn=%s heard_after=%.2fs playback=%.2fs total=%.2fs",
            str(turn_id)[:8], rec.get("client_heard_after_seconds") or 0.0,
            rec.get("playback_seconds") or 0.0, rec.get("total_seconds") or 0.0)
    return rec


def _read_all():
    if not METADATA_JSONL.exists():
        return []
    out = []
    with METADATA_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning("Skipping corrupt metadata line")
    return out


def _write_all(records):
    tmp = METADATA_JSONL.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(METADATA_JSONL)
    # CSV is always regenerated from the JSONL, so its columns can never drift out of sync.
    with METADATA_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def persist_metadata(record):
    try:
        with _meta_lock:
            records = _read_all()
            records.append(dict(record))
            _write_all(records)
    except Exception:
        logging.exception("Could not persist conversation metadata")


def update_metadata(turn_id, fields):
    """Patch an existing record (used when the client reports playback time). Returns it or None."""
    with _meta_lock:
        records = _read_all()
        for r in records:
            if r.get("turn_id") == turn_id:
                r.update(fields)
                finalize_latency(r)
                _write_all(records)
                return r
    return None


def read_metadata(limit=50):
    with _meta_lock:
        return _read_all()[-limit:]


# ---------------------------------------------------------------- the pipeline
def run_turn(conversation, rec, *, audio=None, audio_suffix=".webm", text="", intercept=None):
    """Run one conversation turn and yield events as they happen (a generator).

    Events (dicts):
      {"type": "start", "turn_id"}
      {"type": "stt", "text", "language", "seconds", "backend"}           (voice turns only)
      {"type": "text", "seq", "text"}            a sentence of the reply, as soon as it is known
      {"type": "clip", "seq", "text", "audio", "mime"}   mp3 bytes for that sentence, in order
      {"type": "done", "turn"}   or   {"type": "error", "stage", "error", "turn"}

    `rec` (from new_record) is filled in with every timing and saved + logged at the end.
    `intercept(text)` may return (reply, status) to answer without calling the LLM (exit commands).
    """
    t_req = time.perf_counter()
    since = lambda: time.perf_counter() - t_req  # noqa: E731
    pool = _tts_pool()
    pending = deque()          # (seq, sentence, future, submitted_at) in speaking order
    spoken_parts = []          # cleaned sentences = what is shown and spoken
    stats = {"first_submit": None, "last_ready": None, "tts_fail": 0, "tts_error": ""}
    user_msg = None
    assistant_added = False
    finished = False

    def _finish():
        nonlocal finished
        if finished:
            return
        finished = True
        rec["server_total_seconds"] = since()
        rec["sentences"] = len(spoken_parts)
        if spoken_parts:
            rec["bot_response"] = " ".join(spoken_parts)
        finish_turn(rec)

    def _rollback():
        """Drop the user message if the turn never produced an answer (compare by identity)."""
        if user_msg is not None and not assistant_added:
            for i, m in enumerate(conversation):
                if m is user_msg:
                    del conversation[i]
                    break

    def _queue(sentences, llm_started=None):
        events = []
        for s in sentences:
            spoken = clean_for_speech(s)
            if not spoken:
                continue
            seq = len(spoken_parts)
            spoken_parts.append(spoken)
            if seq == 0 and llm_started is not None:
                rec["llm_first_sentence_seconds"] = time.perf_counter() - llm_started
            events.append({"type": "text", "seq": seq, "text": spoken})
            if stats["tts_fail"] < 2:   # a dead key/quota would otherwise fail every sentence
                now = time.perf_counter()
                if stats["first_submit"] is None:
                    stats["first_submit"] = now
                pending.append((seq, spoken, pool.submit(_tts_clip, spoken), now))
        return events

    def _emit(item):
        seq, sentence, fut, submitted = item
        try:
            res = fut.result()
        except Exception as exc:
            stats["tts_fail"] += 1
            stats["tts_error"] = f"{type(exc).__name__}: {exc}"
            logging.error("TTS failed for turn %s sentence %d: %s", rec["turn_id"][:8], seq, exc)
            return None
        if stats["last_ready"] is None or res["done_at"] > stats["last_ready"]:
            stats["last_ready"] = res["done_at"]
        if not rec["time_to_first_audio_seconds"]:
            rec["time_to_first_audio_seconds"] = since()
            rec["tts_first_audio_seconds"] = res["done_at"] - submitted
        return {"type": "clip", "seq": seq, "text": sentence, "audio": res["audio"], "mime": "audio/mpeg"}

    def _ready():
        out = []
        while pending and pending[0][2].done():
            ev = _emit(pending.popleft())
            if ev:
                out.append(ev)
        return out

    def _fail(stage, exc):
        rec["status"] = f"{stage}_error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        logging.error("Turn %s failed at %s", rec["turn_id"][:8], stage, exc_info=exc)
        _rollback()
        _finish()
        return {"type": "error", "stage": stage, "error": rec["error"], "turn": dict(rec)}

    try:
        yield {"type": "start", "turn_id": rec["turn_id"]}

        # 1. speech to text -----------------------------------------------------------
        if audio is not None:
            rec["audio_kb"] = round(len(audio) / 1024, 1)
            try:
                text, rec["language"], secs, rec["stt_backend"] = transcribe_audio(audio, audio_suffix)
            except Exception as exc:
                yield _fail("stt", exc)
                return
            rec["transcription_seconds"] = secs
            yield {"type": "stt", "text": text, "language": rec["language"],
                   "seconds": round(secs, 4), "backend": rec["stt_backend"]}
        text = (text or "").strip()
        rec["user_input"] = text

        # 2. decide what to say ---------------------------------------------------------
        canned, canned_status = None, "success"
        if not text:
            canned, canned_status = "Sorry, I could not understand. Please try again.", "empty_transcription"
        elif intercept is not None:
            hit = intercept(text)
            if hit:
                canned, canned_status = hit

        if canned is not None:
            for ev in _queue([canned]):
                yield ev
            final_status = canned_status
        else:
            # 3. language model, streamed; sentences go to TTS while the rest is still coming
            user_msg = {"role": "user", "content": text}
            conversation.append(user_msg)
            llm_started = time.perf_counter()
            chunker = SentenceChunker()
            try:
                stream = _open_llm_stream(list(conversation))
                for chunk in stream:
                    delta = _delta_text(chunk)
                    if not delta:
                        continue
                    if not rec["llm_first_token_seconds"]:
                        rec["llm_first_token_seconds"] = time.perf_counter() - llm_started
                    for ev in _queue(chunker.feed(delta), llm_started):
                        yield ev
                    for ev in _ready():
                        yield ev
                for ev in _queue(chunker.flush(), llm_started):
                    yield ev
                if not spoken_parts:
                    for ev in _queue(["Sorry, I could not generate an answer."], llm_started):
                        yield ev
            except Exception as exc:
                yield _fail("llm", exc)
                return
            rec["llm_seconds"] = time.perf_counter() - llm_started
            conversation.append({"role": "assistant", "content": " ".join(spoken_parts)})
            assistant_added = True
            # Keep the system prompt + the last 12 messages (always starts on a user turn).
            if len(conversation) > 13:
                conversation[:] = [conversation[0]] + conversation[-12:]
            final_status = "success"

        # 4. flush the remaining audio, in order -----------------------------------------
        while pending:
            ev = _emit(pending.popleft())
            if ev:
                yield ev

        if stats["first_submit"] is not None and stats["last_ready"] is not None:
            rec["tts_generation_seconds"] = max(0.0, stats["last_ready"] - stats["first_submit"])
        if stats["tts_error"]:
            rec["status"], rec["error"] = "tts_error", stats["tts_error"]
        else:
            rec["status"] = final_status
        _finish()
        yield {"type": "done", "turn": dict(rec)}
    except GeneratorExit:            # the client went away / pressed interrupt
        if rec["status"] == "started":
            rec["status"] = "interrupted"
        raise
    finally:
        for _, _, fut, _ in pending:
            fut.cancel()
        if not finished:
            _rollback()
            _finish()

def _tts_clip(text):
    """Try ElevenLabs first, then fall back to Edge TTS.
    Returns {"audio", "first_byte", "seconds", "done_at"}.
    """
    last_exc = None

    # Primary provider: ElevenLabs
    for attempt in (1, 2):
        started = time.perf_counter()

        try:
            parts, first = [], None

            for part in eleven_client().text_to_speech.convert(
                voice_id=VOICE_ID,
                text=text,
                model_id=TTS_MODEL,
                output_format=TTS_OUTPUT_FORMAT,
            ):
                if part:
                    if first is None:
                        first = time.perf_counter() - started
                    parts.append(part)

            audio = b"".join(parts)

            if not audio:
                raise RuntimeError("ElevenLabs returned no audio")

            done = time.perf_counter()

            return {
                "audio": audio,
                "first_byte": first or 0.0,
                "seconds": done - started,
                "done_at": done,
            }

        except Exception as exc:
            last_exc = exc
            logging.warning(
                "ElevenLabs TTS attempt %d failed: %s",
                attempt,
                exc,
            )

            if attempt == 1:
                time.sleep(0.25)

    # Fallback provider: Edge TTS
    started = time.perf_counter()

    async def generate_edge_audio():
        parts = []
        first = None

        voice = "en-IN-NeerjaNeural"

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
        )

        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data", b"")

                if data:
                    if first is None:
                        first = time.perf_counter() - started
                    parts.append(data)

        audio = b"".join(parts)

        if not audio:
            raise RuntimeError("Edge TTS returned no audio")

        return audio, first

    try:
        audio, first = asyncio.run(generate_edge_audio())
        done = time.perf_counter()

        logging.info("TTS fallback succeeded using Edge TTS")

        return {
            "audio": audio,
            "first_byte": first or 0.0,
            "seconds": done - started,
            "done_at": done,
        }

    except Exception as edge_exc:
        logging.exception("Edge TTS fallback failed")

        raise RuntimeError(
            f"Both TTS providers failed. "
            f"ElevenLabs error: {last_exc}; "
            f"Edge TTS error: {edge_exc}"
        ) from edge_exc
