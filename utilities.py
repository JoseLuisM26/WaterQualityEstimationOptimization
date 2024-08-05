import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import keyboard
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def splitInformation(root, variablePredict):
    df = pd.read_csv(root, sep=",", encoding='latin-1')
    df= df.select_dtypes(include=['float64'])

    xData = df
    xData = xData.dropna()
    yData = xData[variablePredict]
    xData = xData.drop(variablePredict, axis= 1)

    return xData, yData


def pruebaUnitaria(model, scaler, variablePredict):
    rutas_dn = "C:/Users/Jose Luis Medina J/Documents/DOCTORADO/Datos_nuevos_2022.csv"
    dfDatosNuevos = pd.read_csv(rutas_dn, sep =",", encoding="latin-1")
    datos_nuevos_X = dfDatosNuevos
    datos_nuevos_X = datos_nuevos_X.dropna()
    Y_nuevo = datos_nuevos_X[variablePredict]
    X_nuevo = datos_nuevos_X.drop(variablePredict, axis=1)

    xNuevoScaled = scaler.transform(X_nuevo)
    yNuevaPredict = model.predict(xNuevoScaled)
    xNueva = np.arange(len(yNuevaPredict))
    plt.figure(figsize=(10, 6))

# Graficar valores reales vs. predicciones
    plt.plot(xNueva, Y_nuevo, marker='o', linestyle='-', color='blue', label='Valores Reales', alpha =0.5)
    plt.plot(xNueva, yNuevaPredict, marker='o', linestyle='--', color='red', label='Predicciones', alpha=0.5)

    # Etiquetas y título
    plt.xlabel('Número de Filas')
    plt.ylabel('Valores')
    plt.title('Valor Real vs. Predicción por Número de Filas')

    # Leyenda
    plt.legend()

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()


def selectScaler(scalerSelected, Xs, Y):
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(Xs, Y, test_size=test_size)
    if(scalerSelected == 1):
        scaler = preprocessing.RobustScaler()
        scaler = scaler.fit(Xs)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    if(scalerSelected == 2):
        scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(Xs)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    if(scalerSelected ==3):
        scaler = preprocessing.StandardScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

