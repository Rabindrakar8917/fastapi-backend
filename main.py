from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os

app = FastAPI()

# Better Whisper model for Indian languages
model = WhisperModel(
    "small",
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

@app.post("/translate_voice")
async def translate_voice(
    audio: UploadFile = File(...)
):

    audio_path = f"temp_{audio.filename}"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # Detect speech
    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    original_text = ""

    for segment in segments:
        original_text += segment.text + " "

    # Translate detected speech
    translated = GoogleTranslator(
        source='auto',
        target='en'
    ).translate(original_text)

    # Delete temp file
    os.remove(audio_path)

    return {
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "original_text": original_text.strip(),
        "translated_text": translated
    }
