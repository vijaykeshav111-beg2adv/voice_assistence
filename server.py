import base64
import json
import logging
import threading
import uuid

from datetime import date, time

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import core

from appointments import (
    book_appointment,
    get_appointment,
    cancel_appointment,
    reschedule_appointment,
)

from voice_appointments import (
    handle_appointment_text,
    reset_appointment_session,
)
from slots import get_active_doctors, get_available_slots
app = FastAPI(title="Voice Bot")
STATIC = core.BASE_DIR / "static"

_sessions = {}
_sessions_lock = threading.Lock()

logging.basicConfig(level=logging.INFO)
def conversation_for(session_id):
    with _sessions_lock:
        return _sessions.setdefault(session_id, core.new_conversation())


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    stt = (f"groq {core.GROQ_STT_MODEL}" if core.STT_BACKEND == "groq"
           else f"whisper-{core.WHISPER_MODEL} (local)")
    return {
        "llm": core.GROQ_MODEL, "tts": core.TTS_MODEL, "stt": stt,
        "max_record_seconds": core.MAX_WEB_RECORD_SECONDS,
        "missing_keys": core.missing_keys(),
    }


@app.post("/api/warm")
def warm():
    """The browser calls this when recording starts, so the HTTPS connections to Groq and
    ElevenLabs are already open when the user finishes speaking."""
    core.prewarm()
    return {"ok": True}


def _suffix_for(upload: UploadFile):
    ct = (upload.content_type or "").lower()
    for key, ext in (("webm", ".webm"), ("ogg", ".ogg"), ("mp4", ".mp4"), ("m4a", ".m4a"),
                     ("wav", ".wav"), ("mpeg", ".mp3")):
        if key in ct:
            return ext
    return ".webm"


def _line(event):
    """Encode one pipeline event as a line of JSON (NDJSON). Audio bytes become base64."""
    raw_audio = event.get("audio")
    if event.get("type") == "clip":
        event = {k: v for k, v in event.items() if k != "audio"}
        event["audio_b64"] = base64.b64encode(raw_audio).decode()
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/turn")
def turn(
    session_id: str = Form(...),
    audio: UploadFile | None = File(None),
    text: str = Form(""),
    recording_seconds: float = Form(0.0),
):
    """One conversation turn: audio (or typed text) -> STT -> LLM -> TTS.

    The reply is STREAMED as newline-delimited JSON events (see core.run_turn): the first
    spoken sentence reaches the browser while the rest of the answer is still being generated.
    """
    miss = core.missing_keys()
    if miss:
        return JSONResponse({"error": f"{', '.join(miss)} missing in .env", "stage": "config"}, 400)

    audio_bytes = audio.file.read() if audio is not None else None
    suffix = _suffix_for(audio) if audio is not None else ".webm"
    if not audio_bytes:
        audio_bytes = None

    rec = core.new_record(str(uuid.uuid4()), source="voice" if audio_bytes else "text")
    rec["recording_seconds"] = round(max(0.0, min(recording_seconds, 600.0)), 4)
    conv = conversation_for(session_id)

    def stream():
          def appointment_intercept(user_text):
                return handle_appointment_text(session_id, user_text)
      
          for event in core.run_turn(
                conv,
                rec,
                audio=audio_bytes,
                audio_suffix=suffix,
                text=text,
                intercept=appointment_intercept,
            ):
            yield _line(event)

    return StreamingResponse(
        stream(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class Playback(BaseModel):
    playback_seconds: float
    heard_after_seconds: float | None = None


@app.post("/api/turn/{turn_id}/playback")
def playback(turn_id: str, body: Playback):
    """The browser reports how long the reply played and how long after the user stopped
    speaking it first became audible. This completes the turn's totals and is logged."""
    rec = core.report_playback(turn_id, body.playback_seconds, body.heard_after_seconds)
    if rec is None:
        return JSONResponse({"error": "unknown turn"}, 404)
    return {"turn": rec}


@app.get("/api/history")
def history(limit: int = 50):
    return {"turns": core.read_metadata(max(1, min(limit, 500)))}


@app.post("/api/reset")
def reset(session_id: str = Form(...)):
    with _sessions_lock:
        _sessions.pop(session_id, None)

    reset_appointment_session(session_id)
    return {"ok": True}

@app.get("/api/export.csv")
def export_csv():
    if not core.METADATA_CSV.exists():
        return JSONResponse({"error": "no turns recorded yet"}, 404)
    return FileResponse(core.METADATA_CSV, media_type="text/csv", filename="conversation_metadata.csv")


class AppointmentRequest(BaseModel):
    patient_name: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=7, max_length=20)
    doctor_name: str = Field(min_length=1, max_length=100)
    appointment_date: date
    appointment_time: time


class AppointmentChange(BaseModel):
    phone: str = Field(min_length=7, max_length=20)
    appointment_date: date
    appointment_time: time


class AppointmentIdentity(BaseModel):
    phone: str = Field(min_length=7, max_length=20)


@app.post("/api/appointments", status_code=201)
def api_book_appointment(body: AppointmentRequest):
    try:
        result = book_appointment(**body.model_dump())
        return {"success": True, "appointment": result}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to book appointment. Please try again.",
        )


@app.get("/api/appointments/{appointment_id}")
def api_get_appointment(appointment_id: int, phone: str):
    result = get_appointment(appointment_id, phone)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Appointment not found.",
        )

    return {"success": True, "appointment": result}


@app.patch("/api/appointments/{appointment_id}/reschedule")
def api_reschedule(appointment_id: int, body: AppointmentChange):
    try:
        result = reschedule_appointment(
            appointment_id=appointment_id,
            phone=body.phone,
            new_date=body.appointment_date,
            new_time=body.appointment_time,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Appointment not found or no longer active.",
            )

        return {"success": True, "appointment": result}

    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to reschedule appointment.",
        )


@app.patch("/api/appointments/{appointment_id}/cancel")
def api_cancel(appointment_id: int, body: AppointmentIdentity):
    try:
        cancelled = cancel_appointment(appointment_id, body.phone)

        if not cancelled:
            raise HTTPException(
                status_code=404,
                detail="Appointment not found or already cancelled.",
            )

        return {
            "success": True,
            "appointment_id": appointment_id,
            "status": "cancelled",
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to cancel appointment.",
        )
@app.get("/api/doctors")
def api_get_doctors():
    try:
        return {
            "success": True,
            "doctors": get_active_doctors(),
        }
    except Exception:
        logging.exception("Failed to fetch doctors")
        raise HTTPException(
            status_code=500,
            detail="Unable to fetch doctors.",
        )


@app.get("/api/doctors/{doctor_id}/slots")
def api_get_slots(doctor_id: int, date: date):
    try:
        result = get_available_slots(doctor_id, date)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Active doctor not found.",
            )

        return {
            "success": True,
            **result,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        logging.exception("Failed to fetch available slots")
        raise HTTPException(
            status_code=500,
            detail="Unable to fetch available slots.",
        )
if __name__ == "__main__":
    if core.missing_keys():
        print("Warning:", ", ".join(core.missing_keys()), "missing - the UI will show a setup notice.")
    else:
        core.prewarm(force=True)
        if core.STT_BACKEND != "groq":
            print("Loading local Whisper model (first run downloads it)...")
            core.warm_local_whisper()
    print(f"STT: {core.STT_BACKEND} | LLM: {core.GROQ_MODEL} | TTS: {core.TTS_MODEL}")
    print("Open http://localhost:8000   (latency lines are written to logs/voicebot.log)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
