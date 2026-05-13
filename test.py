from deep_translator import GoogleTranslator

text = "to me committee acha"

translated = GoogleTranslator(
    source="auto",
    target="or"
).translate(text)

print(translated)