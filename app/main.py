from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.recommender import Recommender

app = FastAPI()
rec = Recommender("data/books.db")

@app.get("/api/by-title")
def by_title(title: str, n: int = 5):
    return rec.by_title(title, n)

@app.get("/api/by-text")
def by_text(q: str, n: int = 5):
    return rec.by_text(q, n)

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")