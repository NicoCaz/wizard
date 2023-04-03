#Funciones utiles
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


def create_qa_pipeline(model_name):
    """
    Crea una instancia del pipeline de Hugging Face para responder preguntas.
    :param model_name: Nombre del modelo a utilizar.
    :return: Instancia del pipeline de Hugging Face.
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline
