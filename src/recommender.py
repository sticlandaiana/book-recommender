import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_genres(x, menu):
    if pd.isna(x):
        return ""
    g = [s.strip() for s in x.split(",")]
    while g[:len(menu)] == menu:
        g = g[len(menu):]
    g = [s for s in g if s and "Goodreads" not in s]
    return " ".join(dict.fromkeys(s.replace(" ", "_") for s in g))


def prepare_dataframe(csv_path):
    """Citește CSV-ul brut și returnează tabelul curățat."""
    df = pd.read_csv(csv_path).drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["title"]).sort_values("num_ratings", ascending=False)
    df = df.drop_duplicates(subset=["title", "author"]).reset_index(drop=True)

    menu = df["genres"].dropna().iloc[0].split(",")[:29]      # blocul de meniu Goodreads
    df["genres_clean"] = df["genres"].apply(lambda x: clean_genres(x, menu))
    for c in ["description", "author", "series", "title"]:
        df[c] = df[c].fillna("")

    df["main_author"] = df["author"].str.split(",").str[0].str.strip().str.lower()
    df["title_norm"] = (df["title"].str.lower()
                        .str.replace("&", "and", regex=False)
                        .str.replace(r"\(.*?\)", "", regex=True)
                        .str.split(r":|, or ", regex=True).str[0]
                        .str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip())
    df["text"] = (df["title"] + " " + df["author"] + " " + df["series"] + " "
                  + df["genres_clean"] + " " + df["genres_clean"] + " " + df["description"])
    return df


class Recommender:
    def __init__(self, db_path="data/books.db"):
        with sqlite3.connect(db_path) as con:
            df = pd.read_sql("SELECT * FROM books", con)
        for c in ["description", "author", "series", "title", "genres_clean", "text"]:
            df[c] = df[c].fillna("")

        self.df = df
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=20000,
                                     ngram_range=(1, 2), min_df=2)
        self.X = self.tfidf.fit_transform(df["text"])
        self.terms = self.tfidf.get_feature_names_out()
        self.titles = df["title"].str.lower()

    def _explain(self, qvec, j, k=3):
        common = qvec.multiply(self.X[j]).toarray().ravel()
        top = common.argsort()[::-1][:k]
        return [self.terms[t].replace("_", " ") for t in top if common[t] > 0]

    def _top_unique(self, sims, qvec, exclude=None, n=5):
        d = self.df
        seen = set()
        if exclude is not None:
            seen.add((d.title_norm[exclude], d.main_author[exclude]))
        out = []
        for j in sims.argsort()[::-1]:
            key = (d.title_norm[j], d.main_author[j])
            lang = d.language[j]
            if j == exclude or key in seen:
                continue
            if pd.notna(lang) and lang != "English":
                continue
            t = d.title[j].lower()
            if "boxed set" in t or "collection" in t:
                continue
            seen.add(key)
            out.append({
                "id": int(j) + 1,
                "title": d.title[j],
                "author": d.author[j],
                "rating": None if pd.isna(d.avg_rating[j]) else float(d.avg_rating[j]),
                "score": round(float(sims[j]), 3),
                "because": self._explain(qvec, j),
                "description": d.description[j],
            })
            if len(out) == n:
                break
        return out

    def similar(self, idx, n=6):
        qvec = self.X[idx]
        return self._top_unique(cosine_similarity(qvec, self.X).ravel(), qvec, exclude=idx, n=n)
    
    def by_title(self, title, n=5):
        m = self.df.index[self.titles.str.contains(title.lower(), regex=False)]
        if len(m) == 0:
            return []
        idx = m[0]
        qvec = self.X[idx]
        return self._top_unique(cosine_similarity(qvec, self.X).ravel(), qvec, exclude=idx, n=n)

    def by_text(self, query, n=5):
        qvec = self.tfidf.transform([query])
        if qvec.nnz == 0:
            return []
        return self._top_unique(cosine_similarity(qvec, self.X).ravel(), qvec, n=n)