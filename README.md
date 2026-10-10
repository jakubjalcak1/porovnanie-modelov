# Porovnanie modelov: KNN vs. rozhodovací strom

Skript `porovnanie_modelov.py` porovná dva typy klasifikátorov na dátach o emisiách CO2 a vykreslí, ako sa mení trénovacia a testovacia presnosť.

## Dáta

OWID CO2 dataset (Our World in Data), súbor `co2-data.csv`. Súbor nie je súčasťou repozitára. Stiahni ho z <https://github.com/owid/co2-data> a daj ho do rovnakého priečinka ako skript.

- Príznaky (X): `population`, `year`, `gdp`
- Cieľ (y): 1 ak `co2` je nad mediánom, inak 0
- Riadky s NaN v týchto stĺpcoch sa vyhodia
- Rozdelenie: `train_test_split(test_size=0.25, random_state=42, stratify=y)`

## Modely

- KNN: `n_neighbors` = 1, 3, 5, 9, 15
- Rozhodovací strom: `max_depth` = 1, 3, 5, None

## Spustenie

```
pip install pandas scikit-learn matplotlib
python porovnanie_modelov.py
```

Skript vypíše tabuľku 9 modelov zoradenú podľa testovacej presnosti, riadok `NAJLEPSI: ...` a uloží graf `presnost.png`. Pri opakovanom spustení vychádzajú rovnaké čísla (pevné `random_state`).

## Výsledky

Víťaz, interpretácia a celá tabuľka sú v [ZISTENIA.md](ZISTENIA.md).
