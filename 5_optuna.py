import optuna  # Biblioteka do optymalizacji hiperparametrów
from sklearn.datasets import load_iris  # Ładowanie datasetu Iris
from sklearn.model_selection import cross_val_score  # Walidacja krzyżowa
from sklearn.ensemble import RandomForestClassifier  # Algorytm klasyfikacji
from sklearn.model_selection import train_test_split  # Podział danych na zbiór treningowy i testowy
from sklearn.metrics import accuracy_score  # Metryka dokładności

iris = load_iris()
# print(iris)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_sample_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_sample_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return 1.0 - score.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Najlepsz hiperparametry:')
print(study.best_params)

best_accuracy = 1.0 - study.best_value
print(f'\nNajlepsza dokładność walidacji krzyżowej: {best_accuracy:.4f}')

final_model = RandomForestClassifier(
    n_estimators=study.best_params['n_estimators'],
    max_depth=study.best_params['max_depth'],
    min_samples_split=study.best_params['min_sample_split'],
    min_samples_leaf=study.best_params['min_sample_leaf']
)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'\nDokładność na zbioze testowym: {test_accuracy:.4f}')

print(f'\nWażność cech modelu:')
for i, (feature_name, importance) in enumerate(zip(iris.feature_names, final_model.feature_importances_)):
    print(f'{feature_name}: {importance:.4f}')
