"""Command-line voice bot (microphone loop). For the browser UI run: python server.py"""
import io
import logging
import queue
import threading
import time
import uuid

import numpy as np
import pygame
import scipy.io.wavfile as wavfile
import sounddevice as sd

import core

SILENCE_SECONDS = 1.0   # stop recording after this much quiet once you have spoken


def record_audio():
    """Record until you pause (or RECORD_SECONDS at most). Returns (wav_bytes, seconds)."""
    input("\nPress ENTER to start recording...")
    # Everything measured for the turn starts *after* ENTER, so waiting for the human
    # is not counted as latency.
    print(f"Listening (stops when you pause, max {core.RECORD_SECONDS} s)...")
    blocks = queue.Queue()
    frames = []
    noise, heard, last_voice = 0.0, False, 0.0
    t0 = time.perf_counter()
    with sd.InputStream(samplerate=core.SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=int(0.05 * core.SAMPLE_RATE),
                        callback=lambda indata, n, t, s: blocks.put(indata.copy())):
        while True:
            block = blocks.get()
            frames.append(block)
            level = float(np.abs(block.astype(np.int32)).mean())
            now = time.perf_counter() - t0
            if now < 0.4:                      # measure the room noise first
                noise = max(noise, level)
                continue
            if level > max(250.0, noise * 3):
                heard, last_voice = True, now
            if now >= core.RECORD_SECONDS or (heard and now > 1.0 and now - last_voice > SILENCE_SECONDS):
                break
    seconds = time.perf_counter() - t0
    buf = io.BytesIO()
    wavfile.write(buf, core.SAMPLE_RATE, np.concatenate(frames))
    return buf.getvalue(), seconds


class Speaker:
    """Plays the reply's mp3 clips one after another in a background thread, so the next
    sentence can be generated while the current one is speaking."""

    def __init__(self):
        self._queue = queue.Queue()
        self.first_start = None
        self.end = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, mp3_bytes):
        self._queue.put(mp3_bytes)

    def _run(self):
        while True:
            mp3 = self._queue.get()
            if mp3 is None:
                return
            try:
                pygame.mixer.music.load(io.BytesIO(mp3), "mp3")
                if self.first_start is None:
                    self.first_start = time.perf_counter()
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.02)
                self.end = time.perf_counter()
            except Exception:
                logging.exception("Audio playback failed")

    def finish(self):
        """Wait until everything has been spoken. Returns playback seconds (first sound -> last sound)."""
        self._queue.put(None)
        self._thread.join()
        return (self.end - self.first_start) if self.first_start and self.end else 0.0


def exit_intercept(text):
    return ("Goodbye! फिर मिलेंगे।", "exit") if core.is_exit_command(text) else None


def main():
    core.require_keys()
    pygame.mixer.init()
    if core.STT_BACKEND != "groq":
        print("Loading local Whisper speech model...")
        core.warm_local_whisper()
        print("Whisper model loaded.")
    core.prewarm(force=True)
    print("\n===================================")
    print("    HINDI + ENGLISH AI VOICE BOT")
    print("===================================")
    print(f"STT: {core.STT_BACKEND}  LLM: Groq ({core.GROQ_MODEL})  Voice: ElevenLabs ({core.TTS_MODEL})")
    print(f"Metadata: {core.METADATA_CSV}   Timing log: {core.LOG_DIR / 'voicebot.log'}")
    print("Say 'stop voice bot' to exit.")

    conversation = core.new_conversation()
    startup = "Namaste! Your AI voice assistant is ready."
    print("\nBot:", startup)
    try:
        speaker = Speaker()
        audio, _ = core.synthesize_speech(startup)
        speaker.add(audio)
        speaker.finish()
    except Exception:
        logging.exception("Startup speech failed")
        print("Could not generate startup speech. Check ElevenLabs key/credits.")

    try:
        while True:
            rec = None
            try:
                wav_bytes, rec_seconds = record_audio()
                released = time.perf_counter()
                rec = core.new_record(str(uuid.uuid4()))
                rec["recording_seconds"] = rec_seconds
                speaker = Speaker()
                first_bot_line = True
                for ev in core.run_turn(conversation, rec, audio=wav_bytes, audio_suffix=".wav",
                                        intercept=exit_intercept):
                    if ev["type"] == "stt":
                        print(f"Detected language: {ev['language']}\nYou: {ev['text']}")
                    elif ev["type"] == "text":
                        print(("\nBot: " if first_bot_line else "     ") + ev["text"])
                        first_bot_line = False
                    elif ev["type"] == "clip":
                        speaker.add(ev["audio"])
                    elif ev["type"] == "error":
                        print(f"\nCould not complete the {ev['stage']} step: {ev['error']}")
                playback = speaker.finish()
                heard_after = (speaker.first_start - released) if speaker.first_start else None
                final = core.report_playback(rec["turn_id"], playback, heard_after) or rec
                print(
                    "\nLatency (s): "
                    f"record={final['recording_seconds']:.2f}  STT={final['transcription_seconds']:.2f}  "
                    f"LLM first token={final['llm_first_token_seconds']:.2f}  "
                    f"TTS first clip={final['tts_first_audio_seconds']:.2f}  "
                    f"| FIRST AUDIO={final['time_to_first_audio_seconds']:.2f}  "
                    f"all audio ready={final['server_total_seconds']:.2f}  "
                    f"playback={final['playback_seconds']:.2f}  TOTAL={final['total_seconds']:.2f}"
                )
                if rec["status"] == "exit":
                    break
            except KeyboardInterrupt:
                print("\nVoice bot stopped.")
                break
            except Exception:
                logging.exception("Unexpected turn error")
                print("An error occurred. Check logs/voicebot.log")
    finally:
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
