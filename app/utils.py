from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration


ruta_al_modelo = "C:/Users/Nicolas/Desktop/wizard/modelo/modelos_entrenados/"
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained(ruta_al_modelo)

qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device="cpu")
