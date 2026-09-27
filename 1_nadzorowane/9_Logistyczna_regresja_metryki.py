import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def pokaz_metryki(y_test, y_pred):
    # To jest klasyfikacja binarna: 0 = zdrowy, 1 = chory.
    # Poniższe miary dotyczą klasy pozytywnej, czyli osób chorych (outcome=1).
    # TP: chory poprawnie rozpoznany jako chory.
    # FP: zdrowy błędnie rozpoznany jako chory.
    # FN: chory błędnie rozpoznany jako zdrowy.

    # Precision (precyzja) = TP / (TP + FP).
    # Jaka część osób uznanych przez model za chore rzeczywiście jest chora?
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Recall (czułość) = TP / (TP + FN).
    # Jaką część wszystkich rzeczywiście chorych wykrył model?
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    # F1 to średnia harmoniczna precision i recall:
    # F1 = 2 * precision * recall / (precision + recall).
    # Wysokie F1 wymaga jednocześnie wysokiej precyzji i wysokiej czułości.
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Miary mają zakres 0–1; większa wartość oznacza lepszy wynik danej miary.
    # zero_division=0 przyjmuje wynik 0, gdy mianownik jest zerowy.
    # Nie ma uniwersalnego progu określającego, czy model jest dobry:
    # ocena zależy m.in. od kosztu pominięcia chorego i fałszywego alarmu.
    print(f'Precision (precyzja): {precision:.4f}')
    print(f'Recall (czułość):     {recall:.4f}')
    print(f'F1:                  {f1:.4f}')


df = pd.read_csv('diabetes.csv')
print(f'Ile danych: {df.shape}')
print(df.describe().T.to_string())
print('\nLiczba pustych pól:')
print(df.isna().sum())

# W wybranych kolumnach zamień 0 na NA, policz średnią bez braków
# i wpisz średnią tam, gdzie brak wartości.
# Uwaga: przy rzetelnej ocenie modelu średnie należy wyznaczać wyłącznie
# na zbiorze treningowym, a następnie używać ich także w zbiorze testowym.
# Obecny schemat liczy je przed podziałem, więc wykorzystuje informacje z testu.
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
            'bmi', 'diabetespedigreefunction', 'age']:
    df[col] = df[col].replace(0, np.nan)
    mean_ = df[col].mean()
    df[col] = df[col].replace(np.nan, mean_)

print('Po czyszczeniu danych')
print(df.describe().T.to_string())
print(df.isna().sum())

df.to_csv('cukrzyca_po_obrobce.csv', sep=';', index=False)

X = df.iloc[:, :-1]  # Wszystkie kolumny, bez ostatniej.
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

print('\nLogistic Regression')
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# model.score dla LogisticRegression zwraca accuracy (dokładność),
# czyli udział wszystkich poprawnych przewidywań, dla obu klas łącznie.
print(f'Accuracy (dokładność): {model.score(X_test, y_test):.4f}')
print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1])))
pokaz_metryki(y_test, y_pred)

print(f'Zdrowych, ile chorych: {df.outcome.value_counts()}')
print('Zmiana danych')
# 500 zdrowych, 500 chorych.
# Uwaga: sample(n=500) bez zwracania wymaga co najmniej 500 wierszy
# w każdej klasie. Jeśli jest ich mniej, trzeba zmniejszyć n.
# Wyniki na zbiorze o innych proporcjach klas (zwłaszcza precision i F1)
# nie są bezpośrednio porównywalne z wynikami na pierwotnym zbiorze.
df1 = df.query('outcome==0').sample(n=500)
df2 = df.query('outcome==1').sample(n=500)
df3 = pd.concat([df1, df2])

X = df3.iloc[:, :-1]  # Wszystkie kolumny, bez ostatniej.
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

print('\nLogistic Regression po zmianie danych')
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Accuracy (dokładność): {model.score(X_test, y_test):.4f}')
print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1])))
pokaz_metryki(y_test, y_pred)
