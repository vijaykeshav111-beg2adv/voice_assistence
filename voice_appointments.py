
import json
import logging
import os
import re
from datetime import date, datetime, time, timedelta
from threading import Lock

import core
from appointments import book_appointment
from slots import get_active_doctors, get_available_slots


logger = logging.getLogger(__name__)

_pending = {}
_lock = Lock()


# ---------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------

def _get_state(session_id):
    with _lock:
        state = _pending.get(session_id)
        return state.copy() if state is not None else None


def _set_state(session_id, state):
    with _lock:
        if state is None:
            _pending.pop(session_id, None)
        else:
            _pending[session_id] = state.copy()


def reset_appointment_session(session_id):
    """Clear a pending appointment conversation."""
    _set_state(session_id, None)


# ---------------------------------------------------------------------
# Groq LLM fallback: interpretation only, never books an appointment
# ---------------------------------------------------------------------

def _interpret_with_llm(
    text,
    step,
    doctors=None,
    valid_slots=None,
):
    """
    Interpret speech only when deterministic parsing is insufficient.

    Returns a dictionary containing:
        doctor_id: database doctor ID or None
        date: YYYY-MM-DD or None
        time: HH:MM or None
        decision: yes, no, or unclear

    The LLM never performs database writes or confirms a booking.
    Python validates the returned values before using them.
    """
    try:
        doctor_options = [
            {
                "id": int(doctor["id"]),
                "name": str(doctor["name"]),
                "specialization": str(doctor["specialization"]),
            }
            for doctor in (doctors or [])
        ]

        payload = {
            "step": step,
            "today": date.today().isoformat(),
            "user_text": text,
            "doctors": doctor_options,
            "valid_slots": valid_slots or [],
        }

        model = getattr(
            core,
            "GROQ_MODEL",
            os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        )

        system_prompt = (
            "You interpret speech for a medical appointment assistant. "
            "Understand English, Hindi, Hinglish, Devanagari, Roman Hindi, "
            "and common speech-to-text errors. "
            "Return one valid JSON object only with these keys: "
            "doctor_id, date, time, decision. "
            "Use null for fields that cannot be determined. "
            "doctor_id must be an exact ID from the supplied doctor list. "
            "date must be YYYY-MM-DD. "
            "time must be 24-hour HH:MM. "
            "decision must be yes, no, or unclear. "
            "Never invent a doctor, date, or time. "
            "For relative dates, use the supplied current date. "
            "For time interpretation, select only a supplied valid slot "
            "when valid_slots is not empty. "
            "Understand spoken times such as twelve thirty, "
            "saadhe gyarah, and eleven thirty. "
            "For confirmation, only use yes for a clear affirmative "
            "and no for a clear negative. Otherwise use unclear. "
            "The current step determines which field matters."
        )

        client = core.groq_client()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            temperature=0,
            max_tokens=250,
            response_format={"type": "json_object"},
        )

        content = (response.choices[0].message.content or "").strip()

        content = re.sub(r"^```(?:json)?\\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\\s*```$", "", content).strip()

        if not content:
            logger.warning("LLM returned empty content for step=%s", step)
            return {}

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\\{.*\\}", content, flags=re.DOTALL)
            if not match:
                logger.warning("LLM returned non-JSON content for step=%s", step)
                return {}
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("LLM returned malformed JSON for step=%s", step)
                return {}

        if not isinstance(result, dict):
            return {}

        return result

    except Exception:
        logger.warning(
            "LLM appointment interpretation failed for step=%s",
            step,
            exc_info=True,
        )
        return {}


# ---------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------

def _parse_date(text: str) -> date | None:
    value = (text or "").strip().lower()

    if not value:
        return None

    if re.search(r"\b(today|todays)\b", value) or "आज" in value:
        return date.today()

    if any(
        word in value
        for word in (
            "tomorrow",
            "tmrw",
            "कल",
            "आने वाला कल",
        )
    ):
        return date.today() + timedelta(days=1)

    if any(
        word in value
        for word in ("day after tomorrow", "परसों")
    ):
        return date.today() + timedelta(days=2)

    value = value.replace("_", "-")
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(r"\s+", " ", value).strip()

    # YYYY-MM-DD or compact YYYYMMDD.
    match = re.fullmatch(
        r"(\d{4})-?(\d{2})-?(\d{2})",
        value.replace(" ", ""),
    )

    if match:
        try:
            return date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        except ValueError:
            return None

    # DD-MM-YYYY or DD/MM/YYYY.
    match = re.fullmatch(
        r"(\d{1,2})[-/ ](\d{1,2})[-/ ](\d{4})",
        value,
    )

    if match:
        try:
            return date(
                int(match.group(3)),
                int(match.group(2)),
                int(match.group(1)),
            )
        except ValueError:
            return None

    # Month names in English.
    for fmt in (
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    ):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def _parse_phone(text):
    digits = re.sub(r"\D", "", text or "")

    if 7 <= len(digits) <= 15:
        return digits

    return None


# ---------------------------------------------------------------------
# Deterministic yes/no parsing
# ---------------------------------------------------------------------

def _yes(text):
    value = (text or "").strip().lower().strip(" .,!؟?")

    return value in {
        "yes",
        "yeah",
        "yep",
        "yup",
        "y",
        "confirm",
        "confirmed",
        "correct",
        "okay",
        "ok",
        "sure",
        "haan",
        "han",
        "ha",
        "ji",
        "haan ji",
        "yes please",
        "confirm karo",
        "कर दो",
        "हाँ",
        "हां",
        "जी",
        "पक्का",
    }


def _no(text):
    value = (text or "").strip().lower().strip(" .,!؟?")

    return value in {
        "no",
        "nope",
        "nah",
        "n",
        "cancel",
        "cancel it",
        "no thanks",
        "don't",
        "do not",
        "nahi",
        "nahin",
        "nhi",
        "nahi karo",
        "नहीं",
        "मत करो",
        "रद्द करो",
    }


# ---------------------------------------------------------------------
# Doctor selection
# ---------------------------------------------------------------------

def _choice_index(text, count):
    value = (text or "").strip().lower()
    value = value.replace(".", "").replace(",", "")
    value = re.sub(r"\s+", " ", value).strip()

    match = re.fullmatch(
        r"(?:number|option|doctor|doctor number|choice)?\s*(\d+)",
        value,
    )

    if match:
        index = int(match.group(1))

        if 1 <= index <= count:
            return index - 1

    number_words = {
        "one": 1,
        "first": 1,
        "number one": 1,
        "option one": 1,
        "doctor one": 1,
        "two": 2,
        "second": 2,
        "number two": 2,
        "option two": 2,
        "doctor two": 2,
        "three": 3,
        "third": 3,
        "number three": 3,
        "option three": 3,
        "doctor three": 3,
        "एक": 1,
        "पहला": 1,
        "पहली": 1,
        "दो": 2,
        "दूसरा": 2,
        "दूसरी": 2,
        "तीन": 3,
        "तीसरा": 3,
        "तीसरी": 3,
        "नंबर एक": 1,
        "नंबर दो": 2,
        "नंबर तीन": 3,
    }

    number = number_words.get(value)

    if number and 1 <= number <= count:
        return number - 1

    return None


def _normalize_english(text):
    value = (text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _choose_doctor(text, doctors):
    value = (text or "").strip().lower()

    if not doctors:
        return None

    index = _choice_index(value, len(doctors))

    if index is not None:
        return doctors[index]

    normalized = _normalize_english(value)

    aliases = {
        "Dr Aditi Sharma": [
            "aditi sharma",
            "doctor aditi sharma",
            "dr aditi",
            "aditi",
            "अदिति शर्मा",
            "डॉक्टर अदिति शर्मा",
        ],
        "Dr Neha Verma": [
            "neha verma",
            "doctor neha verma",
            "dr neha",
            "neha",
            "निहा वर्मा",
            "नेहा वर्मा",
            "डॉक्टर नेहा वर्मा",
        ],
        "Dr Rahul Mehta": [
            "rahul mehta",
            "doctor rahul mehta",
            "dr rahul",
            "rahul",
            "राहुल मेहता",
            "डॉक्टर राहुल मेहता",
        ],
    }

    for doctor in doctors:
        name = str(doctor["name"])
        candidates = [name, *aliases.get(name, [])]

        for alias in candidates:
            if any("\u0900" <= char <= "\u097f" for char in alias):
                if alias.lower() in value:
                    return doctor
            else:
                normalized_alias = _normalize_english(alias)

                if normalized_alias and (
                    normalized == normalized_alias
                    or normalized_alias in normalized
                ):
                    return doctor

    # Match a unique name token, never guess between multiple doctors.
    matches = []

    for doctor in doctors:
        doctor_name = _normalize_english(
            str(doctor["name"]).replace("Dr ", "")
        )
        tokens = [
            token
            for token in doctor_name.split()
            if len(token) >= 4
        ]

        if any(token in normalized.split() for token in tokens):
            matches.append(doctor)

    return matches[0] if len(matches) == 1 else None


def _format_doctors(doctors):
    if not doctors:
        return "There are no active doctors in the database right now."

    lines = ["These doctors are currently listed:"]

    for index, doctor in enumerate(doctors, start=1):
        lines.append(
            f"{index}. {doctor['name']} - {doctor['specialization']}"
        )

    lines.append("Please say the doctor's name or list number.")

    return " ".join(lines)


def _find_specialty_match(text, doctors):
    value = (text or "").lower()

    specialty_aliases = {
        "cardiology": [
            "cardiology",
            "cardiologist",
            "heart doctor",
            "heart specialist",
            "दिल के डॉक्टर",
            "हृदय",
        ],
        "dermatology": [
            "dermatology",
            "dermatologist",
            "skin doctor",
            "skin specialist",
            "त्वचा",
            "चर्म रोग",
        ],
        "general physician": [
            "general physician",
            "general doctor",
            "physician",
            "gp",
            "जनरल डॉक्टर",
        ],
    }

    matches = []

    for doctor in doctors:
        specialization = str(doctor["specialization"]).lower()
        aliases = specialty_aliases.get(
            specialization,
            [specialization],
        )

        if any(alias in value for alias in aliases):
            matches.append(doctor)

    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------
# Start appointment booking
# ---------------------------------------------------------------------

def _start_booking(session_id, text):
    try:
        doctors = get_active_doctors()
    except Exception:
        logger.exception("Could not load doctors for voice booking")
        return (
            "I'm unable to access the doctor list right now. "
            "Please try again shortly.",
            "appointment_error",
        )

    if not doctors:
        return (
            "There are no active doctors listed in the database right now.",
            "appointment_unavailable",
        )

    doctor = _choose_doctor(text, doctors)

    if doctor is None:
        doctor = _find_specialty_match(text, doctors)

    # Use LLM only if deterministic matching cannot identify a doctor.
    if doctor is None:
        interpreted = _interpret_with_llm(
            text,
            "doctor",
            doctors=doctors,
        )

        requested_id = interpreted.get("doctor_id")

        if requested_id is not None:
            doctor = next(
                (
                    item
                    for item in doctors
                    if str(item["id"]) == str(requested_id)
                ),
                None,
            )

    if doctor is not None:
        _set_state(
            session_id,
            {
                "step": "date",
                "doctor": doctor,
            },
        )

        return (
            f"You selected {doctor['name']}, "
            f"{doctor['specialization']}. "
            "What date would you prefer? Say tomorrow or use YYYY-MM-DD.",
            "appointment_pending",
        )

    _set_state(
        session_id,
        {
            "step": "doctor",
            "doctors": doctors,
        },
    )

    return (_format_doctors(doctors), "appointment_pending")


# ---------------------------------------------------------------------
# Main appointment conversation
# ---------------------------------------------------------------------

def handle_appointment_text(session_id, text):
    """
    Return (reply, status) when handling an appointment conversation.
    Return None for ordinary conversation so the normal LLM handles it.
    """
    text = (text or "").strip()

    if not text:
        return None

    lower = text.lower().strip()
    state = _get_state(session_id)

    if state is None:
        booking_words = (
            "book appointment",
            "book an appointment",
            "book a doctor",
            "appointment",
            "schedule a doctor",
            "doctor appointment",
            "appointment book",
            "doctor se milna",
            "doctor ko dikhana",
            "अपॉइंटमेंट",
            "डॉक्टर को दिखाना",
            "डॉक्टर की बुकिंग",
        )

        if not any(word in lower for word in booking_words):
            return None

        return _start_booking(session_id, text)

    if lower in {
        "stop",
        "quit",
        "exit",
        "forget it",
        "cancel",
        "cancel appointment",
        "छोड़ो",
        "रहने दो",
    }:
        _set_state(session_id, None)
        return ("Okay, I've stopped the appointment booking.", "success")

    step = state["step"]

    # -------------------------------------------------------------
    # Step 1: Choose doctor
    # -------------------------------------------------------------
    if step == "doctor":
        doctors = state["doctors"]
        normalized_request = _normalize_english(text)

        list_requests = (
            "show me available doctors",
            "show available doctors",
            "show me doctors",
            "show doctors",
            "list doctors",
            "available doctors",
            "available doctor",
            "which doctors",
            "what doctors",
            "doctor list",
            "doctors list",
            "tell me the doctors",
            "so available doctors",
            "show available doctor",
        )

        is_list_request = any(
            phrase in normalized_request
            for phrase in list_requests
        )

        # Also recognize short, clear requests such as
        # "doctors please" or "doctors available".
        words = set(normalized_request.split())
        if (
            "doctor" in words or "doctors" in words
        ) and words.intersection({
            "show", "list", "available", "availability",
            "which", "what", "tell", "display", "please",
        }):
            is_list_request = True

        if is_list_request:
            return (
                _format_doctors(doctors),
                "appointment_pending",
            )
        doctor = _choose_doctor(text, doctors)

        if doctor is None:
            doctor = _find_specialty_match(text, doctors)

        if doctor is None:
            interpreted = _interpret_with_llm(
                text,
                "doctor",
                doctors=doctors,
            )

            requested_id = interpreted.get("doctor_id")

            if requested_id is not None:
                doctor = next(
                    (
                        item
                        for item in doctors
                        if str(item["id"]) == str(requested_id)
                    ),
                    None,
                )

        if doctor is None:
            return (
                "I couldn't identify the doctor from that response. "
                "Please say the doctor's name or the number from the list.",
                "appointment_pending",
            )

        state["doctor"] = doctor
        state["step"] = "date"
        _set_state(session_id, state)

        return (
            f"You selected {doctor['name']}, "
            f"{doctor['specialization']}. "
            "What date would you prefer? Say tomorrow or use YYYY-MM-DD.",
            "appointment_pending",
        )

    # -------------------------------------------------------------
    # Step 2: Choose date
    # -------------------------------------------------------------
    if step == "date":
        selected_date = _parse_date(text)

        if selected_date is None:
            interpreted = _interpret_with_llm(text, "date")
            candidate = interpreted.get("date")

            if isinstance(candidate, str):
                try:
                    selected_date = date.fromisoformat(candidate)
                except ValueError:
                    selected_date = None

        if selected_date is None:
            return (
                "Please give the date as YYYY-MM-DD, "
                "for example 2026-11-02, or say tomorrow.",
                "appointment_pending",
            )

        if selected_date < date.today():
            return (
                "That date is in the past. "
                "Please choose today or a future date.",
                "appointment_pending",
            )

        doctor_id = int(state["doctor"]["id"])

        try:
            slots_result = get_available_slots(
                doctor_id,
                selected_date,
            )
        except Exception:
            logger.exception(
                "Appointment slot lookup failed: doctor_id=%s date=%s",
                doctor_id,
                selected_date,
            )
            return (
                "I couldn't check available slots right now. "
                "Please try again.",
                "appointment_error",
            )

        if slots_result is None:
            _set_state(session_id, None)
            return (
                "That doctor is not currently active in the database.",
                "appointment_unavailable",
            )

        available_slots = slots_result.get("slots", [])

        if not available_slots:
            return (
                f"{state['doctor']['name']} has no available slots on "
                f"{selected_date.isoformat()}. Please say another date.",
                "appointment_pending",
            )

        state["date"] = selected_date
        state["slots"] = available_slots
        state["step"] = "slot"
        _set_state(session_id, state)

        readable_slots = ", ".join(available_slots[:12])
        more = (
            " There are additional slots."
            if len(available_slots) > 12
            else ""
        )

        return (
            f"Available times for {state['doctor']['name']} on "
            f"{selected_date.isoformat()} are: {readable_slots}.{more} "
            "Which time would you like? You can say a time such as 10:30.",
            "appointment_pending",
        )

    # -------------------------------------------------------------
    # Step 3: Choose time slot
    # -------------------------------------------------------------
    if step == "slot":
        requested_time = (
            text.strip()
            .lower()
            .replace(".", ":")
        )

        parsed_time = ""

        # Numeric times such as 11:30 or 17:00.
        for fmt in ("%H:%M", "%I:%M %p"):
            try:
                parsed_time = datetime.strptime(
                    requested_time.upper(),
                    fmt,
                ).strftime("%H:%M")
                break
            except ValueError:
                continue

        # Fall back to speech interpretation for "twelve thirty",
        # "saadhe gyarah", and similar phrases.
        if parsed_time not in state["slots"]:
            interpreted = _interpret_with_llm(
                text,
                "time",
                valid_slots=state["slots"],
            )

            candidate_time = interpreted.get("time")

            if isinstance(candidate_time, str):
                try:
                    candidate_time = datetime.strptime(
                        candidate_time,
                        "%H:%M",
                    ).strftime("%H:%M")
                except ValueError:
                    candidate_time = None

            if candidate_time in state["slots"]:
                parsed_time = candidate_time

        if parsed_time not in state["slots"]:
            return (
                "That time isn't in the available slot list. "
                "Please choose one of the available times I just read out.",
                "appointment_pending",
            )

        state["time"] = time.fromisoformat(parsed_time)
        state["step"] = "name"
        _set_state(session_id, state)

        return (
            "What is the patient's full name?",
            "appointment_pending",
        )

    # -------------------------------------------------------------
    # Step 4: Patient name
    # -------------------------------------------------------------
    if step == "name":
        if (
            len(text) < 2
            or len(text) > 100
            or any(character.isdigit() for character in text)
        ):
            return (
                "Please say the patient's full name without numbers.",
                "appointment_pending",
            )

        state["patient_name"] = text
        state["step"] = "phone"
        _set_state(session_id, state)

        return (
            "What phone number should be linked to the appointment?",
            "appointment_pending",
        )

    # -------------------------------------------------------------
    # Step 5: Patient phone
    # -------------------------------------------------------------
    if step == "phone":
        phone = _parse_phone(text)

        if not phone:
            return (
                "That phone number doesn't look valid. "
                "Please say the digits again.",
                "appointment_pending",
            )

        state["phone"] = phone
        state["step"] = "confirm"
        _set_state(session_id, state)

        return (
            f"Please confirm: book {state['doctor']['name']} for "
            f"{state['patient_name']} on {state['date'].isoformat()} "
            f"at {state['time'].strftime('%H:%M')}, using phone number "
            f"ending {phone[-4:]}. Say yes to book or no to stop.",
            "appointment_pending",
        )

    # -------------------------------------------------------------
    # Step 6: Confirm and book
    # -------------------------------------------------------------
    if step == "confirm":
        if _no(text):
            _set_state(session_id, None)
            return (
                "Okay, I haven't booked anything.",
                "success",
            )

        if _yes(text):
            decision = "yes"
        else:
            interpreted = _interpret_with_llm(text, "confirm")
            decision = interpreted.get("decision", "unclear")

        if decision == "no":
            _set_state(session_id, None)
            return (
                "Okay, I haven't booked anything.",
                "success",
            )

        if decision != "yes":
            return (
                "Please clearly say yes to confirm the booking "
                "or no to stop.",
                "appointment_pending",
            )

        try:
            # Re-check the slot immediately before booking.
            latest = get_available_slots(
                int(state["doctor"]["id"]),
                state["date"],
            )

            requested_slot = state["time"].strftime("%H:%M")

            if (
                not latest
                or requested_slot not in latest.get("slots", [])
            ):
                state["step"] = "date"
                state.pop("time", None)
                state.pop("slots", None)
                _set_state(session_id, state)

                return (
                    "Sorry, that slot is no longer available. "
                    "Please choose the date again.",
                    "appointment_pending",
                )

            result = book_appointment(
                patient_name=state["patient_name"],
                phone=state["phone"],
                doctor_name=state["doctor"]["name"],
                appointment_date=state["date"],
                appointment_time=state["time"],
            )

            if not isinstance(result, dict) or result.get("id") is None:
                raise RuntimeError(
                    "Booking function did not return a confirmed appointment ID."
                )

        except ValueError as exc:
            logger.info(
                "Voice appointment booking rejected: %s",
                exc,
            )
            return (str(exc), "appointment_pending")

        except Exception:
            logger.exception("Voice appointment booking failed")
            _set_state(session_id, None)

            return (
                "I couldn't complete the booking because of a database "
                "error. No successful booking has been confirmed. "
                "Please try again.",
                "appointment_error",
            )

        _set_state(session_id, None)

        return (
            "Your appointment is booked successfully. "
            f"Appointment ID {result['id']}. "
            f"Doctor: {result.get('doctor_name', state['doctor']['name'])}. "
            f"Date: {result.get('appointment_date', state['date'])}. "
            f"Time: {result.get('appointment_time', state['time'])}.",
            "success",
        )

    _set_state(session_id, None)

    return (
        "Let's start the appointment booking again.",
        "success",
    )
