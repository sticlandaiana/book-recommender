import sqlite3
from src.recommender import prepare_dataframe

df = prepare_dataframe("data/Goodreadss Books.csv")
with sqlite3.connect("data/books.db") as con:
    df.to_sql("books", con, if_exists="replace", index=False)
print(f"Salvat {len(df)} cărți în data/books.db")