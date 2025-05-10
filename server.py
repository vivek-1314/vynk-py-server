from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("paraphrase-distilroberta-base-v1")

class TextInput(BaseModel):
    text: str

@app.post("/embed")
def get_embedding(input: TextInput):
    embedding = model.encode(input.text).tolist()
    return {"embedding": embedding}