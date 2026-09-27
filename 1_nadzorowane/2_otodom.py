import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('otodom.csv')
print(df.head().to_string())
print(df.describe().T.round(2).to_string())

sns.heatmap(df.iloc[:,1:].corr(), annot=True)
plt.show()

sns.histplot(df.cena)
plt.show()

plt.scatter(df.powierzchnia, df.cena)
plt.show()
q1 = df.describe().loc["25%", 'cena']
q3 = df.describe().loc["75%", 'cena']

df1 = df[ (df.cena >= q1) & (df.cena <= q3)  ]
sns.histplot(df1.cena)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df1.iloc[:, 2:]
y = df1.cena
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
print(f'Wspolczynnik kierunkowy: {model.coef_}')
print( pd.DataFrame  (model.coef_, df1.iloc[:, 2:].columns)   )
print(f'wyraz wolny: {model.intercept_}')
print(f'Wynik: {model.score(X_test, y_test)}')