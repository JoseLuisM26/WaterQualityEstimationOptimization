import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
import os
from keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten
from keras.models import Sequential
import tensorflow as tf
import multiprocessing
from sklearn.svm import SVR, NuSVR, LinearSVR
import math
import joblib
from utilities import selectScaler, splitInformation
num_cores = multiprocessing.cpu_count()


rootDatos = "C:/Users/jose_/Documents/PROYECTOS DOCTORADO/ESTIMACION/outliers/outliers_parametros_P_TOT.csv"
rootDatos1 = "C:/Users/Jose Luis Medina J/Documents/DOCTORADO/TODO DOCTORADO/PROYECTOS/DSR_SINALOA_CRUDOS.csv"
variablePredict = "P_TOT"
df = pd.read_csv(rootDatos1, sep=",", encoding='latin-1')
df= df.select_dtypes(include=['float64'])

Xs, Y = splitInformation(rootDatos1, variablePredict)

X_train_scaled, X_test_scaled, y_train, y_test, scaler = selectScaler(1, Xs, Y)
# Crear el tipo de aptitud y el tipo de individuo en DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Función para evaluar el modelo RandomForestRegressor con R2
def evaluate(individual):
    # Crear y entrenar el modelo Random Forest con los hiperparámetros del individuo
    n_estimators, max_depth, min_samples_split, min_samples_leaf = individual
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
            # Fija la semilla para resultados reproducibles
    )
    model.fit(X_train_scaled, y_train)  # X_train e y_train son tus datos de entrenamiento
    y_pred = model.predict(X_test_scaled)  # X_test es tu conjunto de prueba

    # Calcula el valor de R2
    r2 = r2_score(y_test, y_pred)  # y_test son las etiquetas reales
    if(r2 >=0.90):
        nombre_archivo = 'genetic_modelosV2/modelo_random_forest.joblib'
        joblib.dump(model, nombre_archivo)
    return r2,

# Configuración de DEAP
toolbox = base.Toolbox()
toolbox.register("attr_n_estimators", random.randint, 50, 200)
toolbox.register("attr_max_depth", random.randint, 2, 100)
toolbox.register("attr_min_samples_split", random.randint, 2, 300)
toolbox.register("attr_min_samples_leaf", random.randint, 2, 100) # Rangos de hiperparámetros
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_n_estimators, toolbox.attr_max_depth, toolbox.attr_min_samples_split, toolbox.attr_min_samples_leaf), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=3, up=300, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parámetros del algoritmo genético
population_size = 100
generations = 1000
threshold_r2 = 0.89  # Umbral de R2 deseado

no_change_generations = 50  # Número de generaciones sin cambios significativos en R2
best_r2_history = []
# Crear una población inicial
population = toolbox.population(n=population_size)

# Define estadísticas para rastrear durante la optimización (opcional)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Evolucionar la población con el algoritmo genético
for gen in range(generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

    # Encuentra el mejor individuo en esta generación
    best_ind = tools.selBest(population, k=1)[0]
    best_r2 = best_ind.fitness.values[0]
    best_r2_history.append(best_r2)

    print(f"Generación {gen + 1}: Mejor R2 = {best_r2}")

    # Verifica si se alcanza el umbral de R2 deseado
    if best_r2 >= threshold_r2:
        algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=num_cores * 3, cxpb=0.3, mutpb=0.7, ngen=200,stats=stats, verbose=True)
        print(f"¡Umbral R2 deseado alcanzado ({threshold_r2})!")
        break
    elif gen < generations - 1:
        # Si no se alcanza el umbral y no estamos en la última generación, reinicializa la población
        print("Reinicializando la población...")
        population = toolbox.population(n=population_size)
    
    # Verifica si no ha habido cambios significativos en R2 durante las últimas 'no_change_generations' generaciones
    if len(best_r2_history) >= no_change_generations and all(best_r2 == best_r2_history[-1] for best_r2 in best_r2_history[-no_change_generations:]):
        print(f"No se han producido cambios significativos en R2 durante las últimas {no_change_generations} generaciones. Deteniendo el algoritmo.")
        break

# Mejor individuo después de todas las generaciones
best_individual = tools.selBest(population, k=1)[0]
best_hyperparameters = best_individual[:4]
best_r2 = best_individual.fitness.values[0]
joblib.dump(scaler, 'genetic_modelosV2/modelo_escalador.joblib')
print("Mejor R2 final =", best_r2)
print("Mejores hiperparámetros =", best_hyperparameters)