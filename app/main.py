import sqlite3
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from src.recommender import Recommender

DB = "data/books.db"
app = FastAPI()
rec = Recommender(DB)

# valori pentru filtre, calculate o singură dată
_g = rec.df["genres_clean"].str.split().explode().value_counts()
GENRES = list(_g.index[:30])
LANGS = list(rec.df["language"].dropna().value_counts().index[:10])


def query(sql, params=()):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


# ---------- recomandări ----------
@app.get("/api/by-title")
def by_title(title: str, n: int = 5):
    return rec.by_title(title, n)


@app.get("/api/by-text")
def by_text(q: str, n: int = 5):
    return rec.by_text(q, n)


# ---------- catalog ----------
@app.get("/api/filters")
def filters():
    return {"genres": GENRES, "languages": LANGS}


@app.get("/api/books")
def books(q: str = "", genre: str = "", language: str = "", min_rating: float = 0,
          max_pages: int = 0, sort: str = "popular", page: int = 1, per_page: int = 20):
    where, params = ["1=1"], []
    if q:
        where.append("(title LIKE ? OR author LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if genre:
        where.append("(' ' || genres_clean || ' ') LIKE ?")
        params.append(f"% {genre} %")
    if language:
        where.append("language = ?")
        params.append(language)
    if min_rating:
        where.append("avg_rating >= ?")
        params.append(min_rating)
    if max_pages:
        where.append("num_pages <= ?")
        params.append(max_pages)
    if sort == "rating":
        where.append("num_ratings >= 100")      # evită cărți cu 5.0 și o singură evaluare
    order = {"popular": "num_ratings DESC", "rating": "avg_rating DESC",
             "title": "title ASC"}.get(sort, "num_ratings DESC")

    clause = " AND ".join(where)
    per_page = max(1, min(per_page, 50))
    page = max(1, page)
    total = query(f"SELECT COUNT(*) AS c FROM books WHERE {clause}", params)[0]["c"]
    rows = query(
        f"""SELECT rowid AS id, title, author, avg_rating, num_ratings, num_pages,
                   language, genres_clean
            FROM books WHERE {clause} ORDER BY {order} LIMIT ? OFFSET ?""",
        params + [per_page, (page - 1) * per_page])
    return {"total": total, "page": page, "per_page": per_page, "items": rows}


# ---------- detalii carte ----------
@app.get("/api/books/{book_id}")
def book(book_id: int, n: int = 6):
    rows = query("SELECT rowid AS id, * FROM books WHERE rowid = ?", (book_id,))
    if not rows:
        raise HTTPException(404, "Carte inexistentă")
    b = rows[0]
    b.pop("text", None)
    b["ratings"] = [b.get(f"rated_{i}") for i in range(1, 6)]
    b["similar"] = rec.similar(book_id - 1, n)
    return b


# ---------- statistici ----------
@app.get("/api/stats")
def stats():
    d = rec.df
    r = d["avg_rating"].dropna()
    r = r[r > 0]
    counts, edges = np.histogram(r, bins=np.arange(0, 5.5, 0.5))
    authors = (d["author"].str.split(",").str[0].str.strip()
               .value_counts().head(10))
    genres = _g.head(10)
    langs = d["language"].dropna().value_counts().head(5)
    return {
        "total_books": int(len(d)),
        "rating_hist": {"labels": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(counts))],
                        "values": [int(c) for c in counts]},
        "top_authors": {"labels": list(authors.index), "values": [int(v) for v in authors.values]},
        "top_genres": {"labels": [g.replace("_", " ") for g in genres.index],
                       "values": [int(v) for v in genres.values]},
        "languages": {"labels": list(langs.index), "values": [int(v) for v in langs.values]},
    }


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")