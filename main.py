from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os

app = FastAPI()

# Better Whisper model for Indian languages
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

@app.get("/")
def home():
    return {
        "message": "Backend Working"
    }

# ======================================
# TEXT TRANSLATION
# ======================================

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
async def translate(req: TranslateRequest):

    translated = GoogleTranslator(
        source=req.source_lang,
        target=req.target_lang
    ).translate(req.text)

    return {
        "translated_text": translated
    }

# ======================================
# VOICE TRANSLATION
# ======================================

# @app.post("/translate_voice")
# async def translate_voice(
#     audio: UploadFile = File(...)
# ):

#     audio_path = f"temp_{audio.filename}"

#     with open(audio_path, "wb") as f:
#         f.write(await audio.read())

#     # Detect speech
#     segments, info = model.transcribe(
#         audio_path,
#         beam_size=5
#     )

#     original_text = ""

#     for segment in segments:
#         original_text += segment.text + " "

#     # Translate detected speech
#     translated = GoogleTranslator(
#         source='auto',
#         target='en'
#     ).translate(original_text)

#     # Delete temp file
#     os.remove(audio_path)

#     return {
#         "detected_language": info.language,
#         "language_probability": info.language_probability,
#         "original_text": original_text.strip(),
#         "translated_text": translated
#     }

@app.post("/translate_voice")
async def translate_voice(
    audio: UploadFile = File(...)
):

    audio_path = f"temp_{audio.filename}"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # =========================
    # WHISPER DETECTION
    # =========================

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    detected_language = info.language

    original_text = ""

    for segment in segments:
        original_text += segment.text + " "

    original_text = original_text.strip()

    # =========================
    # LANGUAGE NAME MAP
    # =========================

    language_names = {

        "hi": "Hindi",
        "kn": "Kannada",
        "or": "Odia",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "bn": "Bengali",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "ur": "Urdu",
        "en": "English",

        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ar": "Arabic",
        "ru": "Russian",
        "zh": "Chinese",
    }

    # =========================
    # AUTO TARGET DETECTION
    # =========================

    # Kannada speaker
    if detected_language == "kn":

        target_lang = "hi"

    # Hindi speaker
    elif detected_language == "hi":

        target_lang = "kn"

    # Odia speaker
    elif detected_language == "or":

        target_lang = "kn"

    # Tamil speaker
    elif detected_language == "ta":

        target_lang = "hi"

    # Telugu speaker
    elif detected_language == "te":

        target_lang = "hi"

    # Malayalam speaker
    elif detected_language == "ml":

        target_lang = "hi"

    # English speaker
    elif detected_language == "en":

        target_lang = "hi"

    # DEFAULT
    else:

        target_lang = "en"

    # =========================
    # TRANSLATION
    # =========================

    translated = GoogleTranslator(
        source='auto',
        target=target_lang
    ).translate(original_text)

    # =========================
    # CLEANUP
    # =========================

    os.remove(audio_path)

    # =========================
    # RESPONSE
    # =========================

    return {

        "detected_language":
            detected_language,

        "detected_language_name":
            language_names.get(
                detected_language,
                "Unknown"
            ),

        "language_probability":
            info.language_probability,

        "original_text":
            original_text,

        "translated_text":
            translated,

        "target_language":
            target_lang
    }