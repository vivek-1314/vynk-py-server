from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

class TextInput(BaseModel):
    text: str

@app.post("/embed")
def get_embedding(input: TextInput):
    embedding = model.encode(input.text).tolist()
    return {"embedding": embedding}