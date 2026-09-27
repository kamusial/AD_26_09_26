from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve,
)

# 1. USTAWIENIA — zmieniaj je, żeby eksperymentować.
SEED = 42                     # Stałe losowanie ułatwia odtworzenie wyniku.
TEST_SIZE = 0.20               # 20% danych odkładamy na końcowy test.
CV_FOLDS = 5                   # Liczba części w walidacji krzyżowej.
POKAZ_WYKRESY = False          # True: otwórz okna wykresów na końcu programu.
STYL = 'whitegrid'             # Możesz wybrać też 'white', 'darkgrid', 'ticks'.
PALETA = 'colorblind'          # Możesz wybrać też 'deep', 'muted', 'Set2'.
DPI = 180                     # 300 nadaje się do wydruku.
FORMATY = ('png', 'svg')       # PNG do podglądu, SVG bez utraty jakości.
LICZBA_PRZEDZIALOW = 24        # Liczba słupków histogramu wieku.
MACIERZ_W_PROCENTACH = False   # True: procent w obrębie klasy rzeczywistej.
POBIERZ_PONOWNIE = False       # True: odśwież lokalną kopię danych.

FOLDER = Path(__file__).resolve().parent / 'wyniki_titanic'
URL = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
NUMERYCZNE = ['age', 'sibsp', 'parch', 'fare']
KATEGORIE = ['pclass', 'sex', 'embarked']
CECHY = NUMERYCZNE + KATEGORIE
ETYKIETY = ['Nie przeżył/a', 'Przeżył/a']


def zapisz_wykres(fig, nazwa, opis):
    """Jednakowy wygląd, podpis źródła i eksport wszystkich wykresów."""
    fig.text(0.01, 0.015, f'TITANIC  •  {opis}  •  Źródło: seaborn-data',
             fontsize=9, color='#526173')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for format_ in FORMATY:
        fig.savefig(FOLDER / f'{nazwa}.{format_}', dpi=DPI, bbox_inches='tight')
    if not POKAZ_WYKRESY:
        plt.close(fig)


def pobierz_i_sprawdz():
    """Pobieranie, kontrola struktury i proste reguły czyszczenia."""
    plik = FOLDER / 'titanic_surowe.csv'
    if not plik.exists() or POBIERZ_PONOWNIE:
        print('Pobieram dane z internetu...')
        try:
            with urlopen(URL, timeout=30) as odpowiedz:
                dane = odpowiedz.read()
            plik.write_bytes(dane)
        except OSError as blad:
            raise RuntimeError('Nie udało się pobrać CSV. Sprawdź połączenie z internetem.') from blad
    else:
        print('Korzystam z lokalnej kopii danych:', plik)

    df = pd.read_csv(plik)
    print('\nRozmiar danych (wiersze, kolumny):', df.shape)
    print('\nPierwsze wiersze:\n', df.head().to_string(index=False))
    print('\nTypy kolumn i liczba braków:')
    print(pd.DataFrame({'typ': df.dtypes, 'braki': df.isna().sum()}))
    print('\nStatystyki liczbowe:\n', df.describe().T.round(2).to_string())
    print('\nIdentyczne wiersze:', df.duplicated().sum())
    # Nie usuwamy ich automatycznie: dwie osoby mogą mieć identyczne cechy.
    # Ten CSV nie zawiera identyfikatora, który pozwoliłby to rozstrzygnąć.

    wymagane = set(CECHY + ['survived'])
    if not wymagane.issubset(df.columns):
        raise ValueError(f'Brakuje kolumn: {wymagane - set(df.columns)}')
    df = df[CECHY + ['survived']].copy()
    # Pomijamy m.in. alive: ta kolumna zdradza odpowiedź, czyli survived!
    # Pomijamy też deck (dużo braków) oraz kolumny powielające inne cechy.

    for kol in NUMERYCZNE + ['survived']:
        df[kol] = pd.to_numeric(df[kol], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    for kol in NUMERYCZNE:
        df.loc[df[kol] < 0, kol] = np.nan  # Ujemny wiek lub opłata to błąd.
    # Zera zostają: brak rodzeństwa/dzieci oraz zerowa opłata są możliwe.
    dozwolone = {'pclass': ['1', '2', '3'], 'sex': ['male', 'female'],
                 'embarked': ['S', 'C', 'Q']}
    for kol, wartosci in dozwolone.items():
        df[kol] = df[kol].astype('string').str.strip().replace('', pd.NA)
        df[kol] = df[kol].where(df[kol].isin(wartosci)).astype(object)
        df[kol] = df[kol].where(pd.notna(df[kol]), np.nan)
    zly_cel = ~df['survived'].isin([0, 1])
    print('Usunięte wiersze bez poprawnej etykiety:', zly_cel.sum())
    df = df.loc[~zly_cel].copy()
    df['survived'] = df['survived'].astype(int)
    liczebnosci = df['survived'].value_counts()
    if len(liczebnosci) != 2 or liczebnosci.min() < 10:
        raise ValueError('Za mało przykładów jednej z klas do podziału i walidacji.')
    # Braki cech uzupełni później Pipeline, ucząc się tylko na treningu.
    return df


def analiza_eda(train):
    """EDA = eksploracyjna analiza danych; oglądamy tylko zbiór treningowy."""
    print('\nEDA — udział klas w treningu:')
    print(train['survived'].value_counts(normalize=True).rename('udział').round(3))
    print('\nOdsetek przeżycia według płci i klasy biletu:')
    print(train.groupby(['sex', 'pclass'])['survived'].agg(['count', 'mean']).round(3))
    widok = train.assign(Wynik=train['survived'].map(dict(enumerate(ETYKIETY))))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Kim byli pasażerowie Titanica?', fontsize=21, fontweight='bold')

    braki = train[CECHY].isna().mean().mul(100).sort_values()
    bars = axes[0, 0].barh(braki.index, braki.values, color='#347C98')
    axes[0, 0].bar_label(bars, fmt='%.1f%%', padding=4, fontsize=9)
    axes[0, 0].set(title='Brakujące wartości przed imputacją', xlabel='Udział braków (%)',
                   xlim=(0, max(10, braki.max() * 1.35)))

    sns.countplot(data=widok, x='Wynik', order=ETYKIETY, hue='Wynik',
                  hue_order=ETYKIETY, palette=PALETA, legend=False, ax=axes[0, 1])
    for kontener in axes[0, 1].containers:
        axes[0, 1].bar_label(kontener, padding=3)
    axes[0, 1].set(title='Czy klasy są zrównoważone?', xlabel='', ylabel='Liczba osób')
    axes[0, 1].margins(y=0.15)

    sns.histplot(data=widok, x='age', hue='Wynik', hue_order=ETYKIETY,
                 bins=LICZBA_PRZEDZIALOW, multiple='stack', palette=PALETA,
                 edgecolor='white', linewidth=0.5, ax=axes[1, 0])
    axes[1, 0].set(title='Wiek a wynik — pominięto brakujące wartości',
                   xlabel='Wiek (lata)', ylabel='Liczba osób')

    sns.barplot(data=train, x='pclass', y='survived', hue='sex',
                order=['1', '2', '3'], hue_order=['female', 'male'], palette=PALETA,
                errorbar=None, ax=axes[1, 1])
    axes[1, 1].set(title='Przeżywalność według klasy biletu i płci',
                   xlabel='Klasa biletu', ylabel='Odsetek osób, które przeżyły', ylim=(0, 1))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    handles, _ = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles, ['Kobiety', 'Mężczyźni'], title='Płeć', frameon=False)
    zapisz_wykres(fig, '01_eda', f'Trening: {len(train)} osób; bez uzupełniania braków')

    fig, ax = plt.subplots(figsize=(8, 6))
    # Korelacja pokazuje związek, a nie dowód przyczynowości.
    korelacje = train[NUMERYCZNE + ['survived']].corr()
    sns.heatmap(korelacje, annot=True, fmt='.2f', cmap='vlag', center=0,
                vmin=-1, vmax=1, square=True, linewidths=1, ax=ax)
    ax.set_title('Korelacje cech liczbowych', fontsize=18, fontweight='bold', pad=18)
    zapisz_wykres(fig, '02_korelacje', 'Trening; korelacja Pearsona; dostępne pary danych')


def przygotowanie():
    """Pipeline zamyka czyszczenie i kodowanie w jednej procedurze."""
    liczby = Pipeline([
        ('braki', SimpleImputer(strategy='median')),
        ('skala', StandardScaler()),  # Ważne zwłaszcza dla KNN i regresji.
    ])
    kategorie = Pipeline([
        ('braki', SimpleImputer(strategy='most_frequent')),
        ('kodowanie', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('liczby', liczby, NUMERYCZNE), ('kategorie', kategorie, KATEGORIE),
    ])


def ucz_i_porownaj(X_train, X_test, y_train, y_test):
    algorytmy = {
        'Regresja logistyczna': LogisticRegression(max_iter=1000, random_state=SEED),
        'Drzewo decyzyjne': DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED),
        'Las losowy': RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                              random_state=SEED, n_jobs=1),
        'KNN': KNeighborsClassifier(n_neighbors=15),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    wyniki, predykcje = [], {}
    for nazwa, algorytm in algorytmy.items():
        print(f'\nUczę: {nazwa}...')
        model = Pipeline([('dane', przygotowanie()), ('model', clone(algorytm))])
        # W każdej części CV imputacja i skalowanie uczą się od nowa.
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        predykcje[nazwa] = (pred, proba)
        # Klasa pozytywna: 1 = przeżył/a. Miary poniżej mają zakres 0–1.
        # Accuracy: udział wszystkich poprawnych odpowiedzi.
        # Precision: jaki odsetek przewidzianych ocalałych faktycznie przeżył?
        # Recall: jaką część rzeczywistych ocalałych wykrył model?
        # F1: średnia harmoniczna precision i recall.
        # ROC AUC: zdolność rozróżniania klas przy różnych progach decyzji.
        wyniki.append({
            'Model': nazwa, 'CV F1': cv_f1.mean(), 'CV odch.': cv_f1.std(),
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred, zero_division=0),
            'Recall': recall_score(y_test, pred, zero_division=0),
            'F1': f1_score(y_test, pred, zero_division=0),
            'ROC AUC': roc_auc_score(y_test, proba),
        })
    tabela = pd.DataFrame(wyniki).set_index('Model').sort_values('CV F1', ascending=False)
    print('\nPORÓWNANIE: CV na treningu; pozostałe metryki na wspólnym teście.')
    print(tabela.round(3).to_string())
    tabela.to_csv(FOLDER / 'porownanie_modeli.csv', encoding='utf-8-sig')
    print('\nModel wybrany według średniego F1 w CV:', tabela.index[0])
    print('CV odch. to zmienność między częściami walidacji, nie przedział ufności.')
    # Punkt odniesienia: model zawsze przewidujący najczęstszą klasę.
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    print(f'Accuracy prostego punktu odniesienia: {baseline.score(X_test, y_test):.3f}')
    print('Nie ma uniwersalnego progu „dobrego modelu”; oceniaj kilka miar naraz.')
    print('Wynik dotyczy tego podziału danych. Testu nie używaj do dostrajania modelu.')
    return tabela, predykcje


def wykresy_modeli(tabela, predykcje, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Który model przewiduje najlepiej?', fontsize=21, fontweight='bold')
    t = tabela.iloc[::-1]
    axes[0].barh(t.index, t['CV F1'], xerr=t['CV odch.'], color='#347C98',
                  capsize=4, error_kw={'elinewidth': 1.5})
    axes[0].set(title='Wybór modelu: walidacja krzyżowa', xlabel='F1: średnia ± odchylenie', xlim=(0, 1))
    sns.heatmap(tabela[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']],
                annot=True, fmt='.3f', cmap='Blues', vmin=0, vmax=1,
                linewidths=1, cbar=False, ax=axes[1])
    axes[1].set(title='Ocena na odłożonym zbiorze testowym', ylabel='', xlabel='')
    zapisz_wykres(fig, '03_porownanie', f'{CV_FOLDS} części CV; test: {len(y_test)} osób')

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle('Jakie błędy popełniają modele?', fontsize=21, fontweight='bold')
    normalizacja = 'true' if MACIERZ_W_PROCENTACH else None
    macierze = [confusion_matrix(y_test, pred, labels=[0, 1], normalize=normalizacja)
                 for pred, _ in predykcje.values()]
    maksimum = 1 if MACIERZ_W_PROCENTACH else max(m.max() for m in macierze)
    for ax, nazwa, macierz in zip(axes.flat, predykcje, macierze):
        sns.heatmap(macierz, annot=True, fmt='.1%' if MACIERZ_W_PROCENTACH else 'd',
                    cmap='Blues', vmin=0, vmax=maksimum, cbar=False, square=True,
                    xticklabels=ETYKIETY, yticklabels=ETYKIETY, ax=ax)
        ax.set(title=nazwa, xlabel='Przewidywanie modelu', ylabel='Rzeczywista klasa')
    opis = 'Procenty w wierszach' if MACIERZ_W_PROCENTACH else 'Liczby osób; wspólna skala kolorów'
    zapisz_wykres(fig, '04_macierze_pomylek', opis)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Modele przy różnych progach decyzji', fontsize=21, fontweight='bold')
    kolory = sns.color_palette(PALETA, n_colors=len(predykcje))
    for (nazwa, (_, proba)), kolor in zip(predykcje.items(), kolory):
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        axes[0].plot(fpr, tpr, color=kolor, lw=2, label=f'{nazwa} · AUC {auc:.3f}')
        axes[1].step(rec, prec, where='post', color=kolor, lw=2, label=f'{nazwa} · AP {ap:.3f}')
    # AP (average precision) podsumowuje krzywą precision-recall.
    axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Losowy ranking')
    axes[1].axhline(y_test.mean(), ls='--', color='gray', lw=1, label='Udział klasy pozytywnej')
    axes[0].set(title='Krzywe ROC', xlabel='Odsetek fałszywych alarmów (FPR)', ylabel='Recall (TPR)')
    axes[1].set(title='Krzywe precision–recall', xlabel='Recall', ylabel='Precision')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9)
    zapisz_wykres(fig, '05_roc_precision_recall', 'Test; klasa pozytywna: przeżył/a')


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style=STYL, palette=PALETA, font='DejaVu Sans', font_scale=1.0,
                  rc={'axes.spines.top': False, 'axes.spines.right': False,
                      'axes.titleweight': 'bold', 'grid.alpha': 0.25,
                      'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    df = pobierz_i_sprawdz()
    X_train, X_test, y_train, y_test = train_test_split(
        df[CECHY], df['survived'], test_size=TEST_SIZE, random_state=SEED,
        stratify=df['survived'],  # Zachowaj podobne proporcje klas.
    )
    print(f'\nTrening: {len(X_train)} osób. Test: {len(X_test)} osób.')
    analiza_eda(X_train.assign(survived=y_train))

    # Osobno zapisujemy przykładowe dane treningowe po całej transformacji.
    # Transformator dopasowujemy tylko do treningu; modele w CV mają własne.
    procesor = przygotowanie()
    przetworzone = procesor.fit_transform(X_train)
    czyste = pd.DataFrame(przetworzone, columns=procesor.get_feature_names_out(), index=X_train.index)
    czyste['survived'] = y_train
    print('\nBraki po przetworzeniu treningu:', czyste.isna().sum().sum())
    czyste.to_csv(FOLDER / 'trening_po_przetworzeniu.csv', index=False, encoding='utf-8-sig')

    tabela, predykcje = ucz_i_porownaj(X_train, X_test, y_train, y_test)
    wykresy_modeli(tabela, predykcje, y_test)
    print('\nGotowe! Wykresy i pliki CSV zapisano w:', FOLDER)
    if POKAZ_WYKRESY:
        plt.show()


if __name__ == '__main__':
    main()
