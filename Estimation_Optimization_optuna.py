import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from utilities import splitInformation, selectScaler, pruebaUnitaria
from optuna.visualization import plot_optimization_history, plot_slice
import matplotlib.pyplot as plt

#wandb.init(project="nombre-del-proyecto", entity="nombre-de-la-entidad")
rootDatos = "C:/Users/jose_/Documents/PROYECTOS DOCTORADO/ESTIMACION/outliers/outliers_parametros_P_TOT.csv"
rootDatos1 = "C:/Users/Jose Luis Medina J/Documents/DOCTORADO/TODO DOCTORADO/PROYECTOS/DSR_SINALOA_CRUDOS_2.csv"
variablePredict = "N_TOTK"
Xs, Y = splitInformation(rootDatos1, variablePredict)
X_train_scaled, X_test_scaled, y_train, y_test, scaler = selectScaler(1, Xs, Y)
rmse_nuevo = 0.0


def objective(trial):
    # Definir el espacio de hiperparámetros
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 10, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_int("max_features", 1, 15)

    # Crear el modelo con los hiperparámetros sugeridos
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    # Realizar validación cruzada y devolver el promedio del score R^2
    r2_scores = cross_val_score(model, X_train_scaled, y_train, n_jobs=-1, cv=5, scoring='r2')
    return np.mean(r2_scores)

# Crear un estudio Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Mejores hiperparámetros encontrados
print('Mejores hiperparámetros:', study.best_params)

plt.plot([trial.value for trial in study.trials])
plt.xlabel("Trial")
plt.ylabel("R2 Score")
plt.title("Optuna Optimization Progress")
plt.show()
study.trials_dataframe().to_csv("optuna_study_results.csv")