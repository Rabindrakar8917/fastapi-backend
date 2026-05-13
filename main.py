# from fastapi import FastAPI
# from pydantic import BaseModel
# from deep_translator import GoogleTranslator

# app = FastAPI()

# @app.get("/")
# def home():
#     return {
#         "message": "Backend Working"
#     }

# class TranslateRequest(BaseModel):
#     text: str
#     source_lang: str
#     target_lang: str

# @app.post("/translate")
# async def translate(req: TranslateRequest):

#     translated = GoogleTranslator(
#         source=req.source_lang,
#         target=req.target_lang
#     ).translate(req.text)

#     return {
#         "translated_text": translated
#     }



# from fastapi import FastAPI, UploadFile, File
# from faster_whisper import WhisperModel
# from deep_translator import GoogleTranslator

# app = FastAPI()

# model = WhisperModel("base")

# @app.get("/")
# def home():
#     return {
#         "message": "Backend Working"
#     }

# @app.post("/translate_voice")
# async def translate_voice(
#     audio: UploadFile = File(...)
# ):

#     audio_path = f"temp_{audio.filename}"

#     with open(audio_path, "wb") as f:
#         f.write(await audio.read())

#     segments, info = model.transcribe(audio_path)

#     original_text = ""

#     for segment in segments:
#         original_text += segment.text

#     translated = GoogleTranslator(
#         source='auto',
#         target='en'
#     ).translate(original_text)

#     return {
#         "language": info.language,
#         "original_text": original_text,
#         "translated_text": translated
#     }

from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os

app = FastAPI()

# Better model for Indian languages
model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

@app.get("/")
def home():
    return {
        "message": "Backend Working"
    }

@app.post("/translate")
async def translate_voice(
    audio: UploadFile = File(...)
    ):

    audio_path = f"temp_{audio.filename}"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # AUTO DETECT LANGUAGE
    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    original_text = ""

    for segment in segments:
        original_text += segment.text + " "

    # TRANSLATE TO ENGLISH
    translated = GoogleTranslator(
        source='auto',
        target='en'
    ).translate(original_text)

    # DELETE TEMP FILE
    os.remove(audio_path)

    return {
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "original_text": original_text.strip(),
        "translated_text": translated
    }