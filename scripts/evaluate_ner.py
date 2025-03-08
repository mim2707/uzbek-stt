from transformers import pipeline

ner_pipeline = pipeline("ner", model="../models/bert-uzbek-ner", tokenizer="bert-base-multilingual-cased")

text = "Toshkent shahrida joylashgan O‘zbekiston Respublikasi Prezidenti Shavkat Mirziyoyev 2023-yil 10-martda yangi qonun imzoladi."

ner_results = ner_pipeline(text)
print(ner_results)