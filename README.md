# Big Data — TP

Enzo Caramella — identifiant 24021136

| TP | Sujet | Dossier |
|---|---|---|
| TP1 | Comptage de mots, quatre méthodes : Python séquentiel, pseudo-MapReduce Python, Hadoop MapReduce, PySpark | [`TP1/`](TP1/) |
| TP2 | Accidents de la route (Australie-Méridionale, 2012–2019) avec PySpark : reproduction d'un article et analyse personnelle | [`TP2/`](TP2/) |
| TP3 | Streaming, deux études de cas Kaggle : popularité des morceaux Spotify (small data, scikit-learn + PySpark) et recommandation de films Netflix sur 100 M de notes (big data, Spark ALS) | [`TP3_ID24021136/`](TP3_ID24021136/) |

Chaque dossier contient un `README.md` qui décrit le TP et explique comment le reproduire.

## Environnement commun

Python 3.12, PySpark 4.2, un JDK 17 ou 21, et Docker pour le mini-cluster Hadoop du TP1. Un environnement virtuel
`.venv/` à la racine (ignoré par git) peut servir à tous les TP :

```bash
pip install jupyter pandas matplotlib numpy pyspark seaborn scikit-learn xgboost joblib
```

Les notebooks se lancent depuis leur propre dossier (`TP1/`, `TP2/`…) : leurs chemins sont relatifs à ce dossier.
