import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

# 1. Wczytanie tych samych danych: dwie pierwsze cechy zbioru Iris.
iris = load_iris()
X = iris.data[:, :2]

print('Dane załadowane:')
print(f'Liczba próbek: {X.shape[0]}')
print(f'Cechy: {iris.feature_names[:2]}')

# 2. Ustawienie parametrów i uruchomienie DBSCAN.
# eps: promień sąsiedztwa punktu.
# min_samples: minimum punktów w sąsiedztwie (łącznie z samym punktem),
# aby punkt był punktem rdzeniowym, od którego można rozszerzać klaster.
# metric: sposób obliczania odległości między punktami.
dbscan = DBSCAN(eps=0.3, min_samples=3, metric='euclidean')
etykiety = dbscan.fit_predict(X)

# 3. Podsumowanie wyników. Etykieta -1 oznacza szum.
numery_klastrow = sorted(set(etykiety) - {-1})
liczba_szumu = np.sum(etykiety == -1)

print(f'\nLiczba klastrów: {len(numery_klastrow)}')
print(f'Liczba punktów szumu: {liczba_szumu}')

for i in numery_klastrow:
    liczba = np.sum(etykiety == i)
    print(f'Klaster {i + 1}: {liczba} próbek')

# 4. Rysowanie każdego klastra innym kolorem.
plt.figure(figsize=(8, 6))
for i in numery_klastrow:
    punkty = X[etykiety == i]
    plt.scatter(punkty[:, 0], punkty[:, 1],
                label=f'Klaster {i + 1}', alpha=0.7, s=50)

# Punkty szumu rysujemy osobno jako czarne krzyżyki.
if liczba_szumu > 0:
    szum = X[etykiety == -1]
    plt.scatter(szum[:, 0], szum[:, 1],
                c='black', marker='x', label='Szum', s=60)

plt.title(f'DBSCAN (eps={dbscan.eps}, min_samples={dbscan.min_samples})')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
