from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


# 1. Pobranie danych: 178 próbek wina i 13 cech chemicznych.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
print('Pobieranie danych Wine...')
try:
    with urlopen(url, timeout=30) as odpowiedz:
        dane = np.loadtxt(odpowiedz, delimiter=',')
except (OSError, ValueError) as blad:
    raise SystemExit(f'Nie udało się pobrać lub odczytać danych: {blad}')

# Pierwsza kolumna zawiera prawdziwą klasę, pozostałe kolumny to cechy.
# Prawdziwych klas NIE przekazujemy do algorytmów klastrowania.
prawdziwe_klasy = dane[:, 0].astype(int)
X = dane[:, 1:]
print(f'Liczba próbek: {X.shape[0]}, liczba cech: {X.shape[1]}')


# 2. Standaryzacja: każda cecha ma średnią 0 i odchylenie standardowe 1.
# Dzięki temu cecha o dużych liczbach nie dominuje w obliczaniu odległości.
X_skala = StandardScaler().fit_transform(X)


# 3. K-means: dla tego przykładu przyjmujemy 3 klastry.
# Wiemy z opisu zbioru, że zawiera on 3 klasy; to jawne założenie przykładu.
# n_init=10: 10 prób z różnymi początkowymi centroidami.
# random_state=42: umożliwia powtarzanie wyniku.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
etykiety_kmeans = kmeans.fit_predict(X_skala)


# 4. DBSCAN: sam wyznacza liczbę klastrów, a szum oznacza jako -1.
# eps=2.2: promień sąsiedztwa w przestrzeni 13 standaryzowanych cech.
# min_samples=5: minimum punktów w sąsiedztwie (wliczając badany punkt),
# aby punkt był rdzeniowy i umożliwiał rozszerzanie klastra.
# Są to przykładowe parametry do eksperymentów, a nie ustawienia optymalne.
dbscan = DBSCAN(eps=2.2, min_samples=5, metric='euclidean')
etykiety_dbscan = dbscan.fit_predict(X_skala)


# 5. Porównanie wyników obu algorytmów.
wyniki = {'K-means': etykiety_kmeans, 'DBSCAN': etykiety_dbscan}

for nazwa, etykiety in wyniki.items():
    klastry = sorted(set(etykiety) - {-1})
    bez_szumu = etykiety != -1
    liczba_szumu = np.sum(~bez_szumu)

    print(f'\n{nazwa}:')
    print(f'Liczba klastrów: {len(klastry)}')
    print(f'Szum: {liczba_szumu} próbek ({liczba_szumu / len(X):.1%})')
    for i in klastry:
        print(f'  Klaster {i + 1}: {np.sum(etykiety == i)} próbek')

    # ARI mierzy zgodność z prawdziwymi klasami niezależnie od numerów klastrów.
    # 1 = identyczny podział, około 0 = zgodność na poziomie losowym.
    # Liczymy dla WSZYSTKICH próbek; szum DBSCAN jest traktowany jako jedna grupa.
    ari = adjusted_rand_score(prawdziwe_klasy, etykiety)
    print(f'ARI (wszystkie próbki): {ari:.3f}')

    # Silhouette ocenia zwartość i oddzielenie klastrów: od -1 do 1.
    # Wymaga co najmniej 2 klastrów i mniej klastrów niż ocenianych próbek.
    # Pomijamy szum, dlatego DBSCAN i K-means mogą oceniać różne zbiory punktów.
    if 2 <= len(klastry) < np.sum(bez_szumu):
        wynik = silhouette_score(X_skala[bez_szumu], etykiety[bez_szumu])
        print(f'Silhouette (bez szumu): {wynik:.3f}')
    else:
        print('Silhouette: brak możliwości obliczenia dla tego podziału.')

print('\nWyższe ARI oznacza większą zgodność z prawdziwymi klasami.')
print('Silhouette porównuj razem z odsetkiem szumu: oceniane próbki mogą się różnić.')
print('To porównanie konkretnych ustawień, a nie ogólny ranking algorytmów.')


# 6. PCA służy TYLKO do wykresów. Klastrowanie wykonaliśmy na 13 cechach.
# Te same współrzędne 2D wykorzystujemy na wszystkich trzech wykresach.
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_skala)
zachowana_wariancja = np.sum(pca.explained_variance_ratio_)

podzialy = {'Prawdziwe klasy': prawdziwe_klasy - 1, **wyniki}
fig, osie = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for os, (nazwa, etykiety) in zip(osie, podzialy.items()):
    for i in sorted(set(etykiety)):
        punkty = X_2d[etykiety == i]
        if i == -1:
            os.scatter(punkty[:, 0], punkty[:, 1],
                       c='black', marker='x', label='Szum', s=40)
        else:
            os.scatter(punkty[:, 0], punkty[:, 1],
                       label=f'Grupa {i + 1}', alpha=0.7, s=35)
    os.set_title(nazwa)
    os.set_xlabel('Pierwsza składowa PCA')
    os.set_ylabel('Druga składowa PCA')
    os.legend()
    os.grid(True, alpha=0.3)

# Numery i kolory grup na różnych wykresach nie oznaczają tej samej klasy.
fig.suptitle(f'Wine — porównanie podziałów (PCA zachowuje {zachowana_wariancja:.1%} wariancji)')
plt.tight_layout()
plt.show()
