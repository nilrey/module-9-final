import pickle

# from gensim.models import Word2Vec
from pathlib import Path

import pandas as pd

# Локальная загрузка моделей (вам необходимо будет это сделать в финальном решении при помощи S3).
with open(Path("./ranker.pkl"), "rb") as f:
    ranker = pickle.load(f)

# Локальная работа с базой данных (условная, конечно). Аналогично требуется сделать это при помощи S3.
item_features = pd.read_parquet(Path("./item_features.parquet"))
