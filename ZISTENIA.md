# Zistenia – porovnanie KNN a rozhodovacieho stromu

Dáta: OWID CO2 dataset (`co2-data.csv`), po vyčistení (bez NaN v `population`, `year`, `gdp`, `co2`).
Príznaky: `population`, `year`, `gdp`. Cieľ: 1 ak `co2` je nad mediánom, inak 0.
Rozdelenie: `test_size=0.25`, `random_state=42`, `stratify=y`.

## Víťaz

**Rozhodovací strom, `max_depth=None`** – testovacia presnosť **0.926** (trénovacia 1.000).

Pre porovnanie, najlepší KNN je `n_neighbors=15` s testovacou presnosťou 0.863.

## Zistenia

Najvyššiu trénovaciu presnosť (1.000) majú KNN s k=1 a strom bez obmedzenia hĺbky, no na testovacích dátach majú nižšiu: KNN s k=1 má 0.844 (rozdiel 0.156), strom 0.926 (rozdiel 0.074). To znamená, že si trénovacie dáta skoro „zapamätali" a časť presnosti pri nových dátach strácajú, teda overfitting. Pri KNN sa so zvyšujúcim k trénovacia presnosť znižuje a testovacia rastie (z 0.844 na 0.863), takže väčšie k model vyhladzuje a lepšie zovšeobecňuje. Pri strome naopak najlepšie vyšiel ten najhlbší, hoci má train 1.000, takže vysoký train sám o sebe nie je zlý, rozhoduje testovacia presnosť. Veľmi plytký strom (`max_depth=1`) je naopak slabý na oboch stranách (train 0.881, test 0.870), čo je opak overfittingu, model je príliš jednoduchý. KNN je pravdepodobne slabší aj preto, že sa nepoužilo škálovanie: `population` a `gdp` majú obrovské hodnoty, takže vzdialenosť určujú skoro len ony a `year` takmer nič, strom týmto problémom netrpí.

## Tabuľka výsledkov

| model | parameter | train_acc | test_acc |
|-------|-----------|-----------|----------|
| tree  | None | 1.000 | 0.926 |
| tree  | 5    | 0.907 | 0.893 |
| tree  | 3    | 0.893 | 0.883 |
| tree  | 1    | 0.881 | 0.870 |
| knn   | 15   | 0.886 | 0.863 |
| knn   | 9    | 0.889 | 0.858 |
| knn   | 5    | 0.905 | 0.854 |
| knn   | 3    | 0.922 | 0.850 |
| knn   | 1    | 1.000 | 0.844 |

Pri mediánovom rozdelení je základná hodnota (tipovanie väčšinovej triedy) približne 0.5, takže všetky modely sú výrazne lepšie ako náhoda.
