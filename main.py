from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Backend Working"
    }

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
