import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

k_values = [1, 3, 5, 9, 15]
depth_values = [1, 3, 5, None]
vysledky = []

data = pd.read_csv("co2-data.csv", sep=",")

data_ciste = data.dropna(subset=["population", "year", "gdp", "co2"])

X = data_ciste[["population", "year", "gdp"]]
y = (data_ciste["co2"] > data_ciste["co2"].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

for value in k_values:
    model = KNeighborsClassifier(n_neighbors=value)
    model.fit(X_train, y_train)
    model_trening = model.predict(X_train)
    model_test = model.predict(X_test)
    trening_acc = accuracy_score(y_train, model_trening)
    test_acc = accuracy_score(y_test, model_test)
    vysledky.append({
        "model": "knn",
        "parameter": value,
        "train_acc": trening_acc,
        "test_acc": test_acc,
    })

for value in depth_values:
    model = DecisionTreeClassifier(max_depth=value, random_state=42)
    model.fit(X_train, y_train)
    model_trening = model.predict(X_train)
    model_test = model.predict(X_test)
    trening_acc = accuracy_score(y_train, model_trening)
    test_acc = accuracy_score(y_test, model_test)
    vysledky.append({
        "model": "tree",
        "parameter": value,
        "train_acc": trening_acc,
        "test_acc": test_acc,
    })

df = pd.DataFrame(vysledky)

df_zoradene = df.sort_values("test_acc",ascending=False)

print(df_zoradene.to_string(index=False))

najlepsi = df_zoradene.iloc[0]
nazov_parametra = "n_neighbors" if najlepsi["model"] == "knn" else "max_depth"

print(f"\nNAJLEPSI: {najlepsi['model']} ({nazov_parametra}={najlepsi['parameter']}), test_acc = {najlepsi['test_acc']:.3f}")

knn_df = df[df["model"] == "knn"]
tree_df = df[df["model"] == "tree"]

fig, axes = plt.subplots(1,2)

axes[0].plot(range(len(k_values)), knn_df["train_acc"], color="red", label="Trening")
axes[0].plot(range(len(k_values)), knn_df["test_acc"], color="green", label="Test")
axes[0].set_title("KNN modely")
axes[0].set_xlabel("k hodnoty")
axes[0].set_ylabel("Presnosť")
axes[0].set_xticks(range(len(k_values)))
axes[0].set_xticklabels(k_values)
axes[0].legend()

axes[1].plot(range(len(depth_values)), tree_df["train_acc"], color="red", label="Trening")
axes[1].plot(range(len(depth_values)), tree_df["test_acc"], color="green", label="Test")
axes[1].set_title("Rozhodovací strom")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("Presnosť")
axes[1].set_xticks(range(len(depth_values)))
axes[1].set_xticklabels([str(h) for h in depth_values])
axes[1].legend()

plt.tight_layout()
plt.savefig("presnost.png")