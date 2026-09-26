import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions


# 1. Wczytanie danych
df = pd.read_csv("iris.csv")

species = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}
species_names = list(species)

df["class_value"] = df["class"].map(species)

print("Liczba przykładów w klasach:")
print(df["class"].value_counts())

print("\nKlasy numeryczne:")
print(df["class_value"].value_counts())


# 2. Wszystkie CZTERY cechy, w ustalonej kolejności
feature_names = [
    "sepallength",
    "sepalwidth",
    "petallength",
    "petalwidth",
]

# Tablice NumPy używane zarówno przy treningu, jak i rysowaniu
X = df[feature_names].to_numpy(dtype=float)
y = df["class_value"].to_numpy(dtype=int)

sample = np.array([5.6, 3.2, 5.2, 1.45])

print("\nWymiary X:", X.shape)  # Dla 150 przykładów: (150, 4)


# 3. Podział danych i trening
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = DecisionTreeClassifier(
    max_depth=20,
    min_samples_split=20,
    random_state=42,
)

model.fit(X_train, y_train)


# 4. Ocena modelu
y_pred = model.predict(X_test)

print(f"\nDokładność: {model.score(X_test, y_test):.2%}")

print("\nMacierz pomyłek:")
print("Wiersze: klasy rzeczywiste, kolumny: przewidywane")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=[0, 1, 2]),
    index=species_names,
    columns=species_names,
))

print("\nWażność cech:")
print(pd.DataFrame(
    {"ważność": model.feature_importances_},
    index=feature_names,
))

sample_class = int(model.predict(sample.reshape(1, -1))[0])
print("\nPrzewidywana klasa próbki:", species_names[sample_class])


# 5. Dwa wykresy wszystkich obserwacji i nowej próbki
for i, j in [(0, 1), (2, 3)]:
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=feature_names[i],
        y=feature_names[j],
        hue="class",
    )

    plt.scatter(
        sample[i],
        sample[j],
        color="red",
        marker="*",
        s=200,
        edgecolors="black",
        label="Nowa próbka",
        zorder=5,
    )

    plt.legend()
    plt.tight_layout()


# 6. Przekrój granic decyzyjnych modelu z CZTEREMA cechami
plt.figure(figsize=(9, 6))

plot_decision_regions(
    X=X,  # Nadal przekazujemy wszystkie cztery kolumny!
    y=y,
    clf=model,

    # Osie: petallength i petalwidth
    feature_index=[2, 3],

    # Stałe wartości pozostałych cech
    filler_feature_values={
        0: sample[0],  # sepallength = 5.6
        1: sample[1],  # sepalwidth = 3.2
    },

    # Pokaż obserwacje znajdujące się blisko przekroju.
    # Te zakresy filtrują punkty, nie zmieniają granic.
    filler_feature_ranges={
        0: 0.3,
        1: 0.3,
    },

    legend=2,
)

plt.scatter(
    sample[2],
    sample[3],
    color="red",
    marker="*",
    s=200,
    edgecolors="black",
    label="Nowa próbka",
    zorder=5,
)

plt.xlabel("petallength")
plt.ylabel("petalwidth")
plt.title("Przekrój: sepallength = 5.6, sepalwidth = 3.2")
plt.legend(loc="upper left")
plt.tight_layout()

plt.show()
