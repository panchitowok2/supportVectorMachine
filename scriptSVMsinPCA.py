"""implementacion de SVM"""
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

class SVM:
    
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, cantidad_de_iteraciones=1000):

        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.cantidad_de_iteraciones = cantidad_de_iteraciones
        self.vector_hiperplano = None
        self.escalar_hiperplano = None
        

    def _inicializar_variables(self,matriz_datos):
        cantidad_de_caracteristicas = len(matriz_datos[0])
        self.vector_hiperplano = np.array([0.0] * cantidad_de_caracteristicas)
        self.escalar_hiperplano = 0

    def _satisface_restriccion(self, instancia, posicion):
        #determina si la instancia esta correctamente clasificada mediante la perdida de bisagra o Hinge loss
        linear_model = np.dot(instancia, self.vector_hiperplano) + self.escalar_hiperplano
       
        return self.arreglo_de_resultados[posicion] * linear_model >= 1
    
    def _calcular_gradientes_y_actualizar_variables(self, constrain, instacia_actual, posicion):
        #calculo los gradientes en base a si se clasifico correctamente la instancia
        if constrain:
            gradiente_vector = self.lambda_param * self.vector_hiperplano
            gradiente_escalar = 0
        else:
            gradiente_vector = self.lambda_param * self.vector_hiperplano - np.dot(self.arreglo_de_resultados[posicion], instacia_actual)
            gradiente_escalar = - self.arreglo_de_resultados[posicion]
       
        self.vector_hiperplano -= self.learning_rate * gradiente_vector
        self.escalar_hiperplano -= self.learning_rate * gradiente_escalar
            
    
   

    def entrenamiento(self, matriz_de_datos, arreglo_de_resultados):
        """rf"""
        self._inicializar_variables(matriz_de_datos)
        self.arreglo_de_resultados = arreglo_de_resultados

        for _ in range(self.cantidad_de_iteraciones):
            for posicion in range(len(matriz_de_datos)):
                instancia_actual = matriz_de_datos[posicion]
                constrain = self._satisface_restriccion(instancia_actual, posicion)
                self._calcular_gradientes_y_actualizar_variables(constrain, instancia_actual, posicion)

    def predecir(self, matriz_datos):
        """ds"""
        estimacion = np.dot(matriz_datos, self.vector_hiperplano) + self.escalar_hiperplano
        prediccion = np.sign(estimacion)
        return prediccion
    
    # Cargamos el conjunto de datos de incendio
data =  pd.read_csv('dataset.csv', sep=",")

# Reemplazar los valores del campo categorico a numerico para el algoritmo
# Reemplazar los valores utilizando el método replace
data['Classes'] = data['Classes'].replace({'fire': 1, 'not fire': -1})

# separamos features del campo categorico que define la clase
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# dividimos el codigo en test de entrenamiento y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Entrenamos el modelo
clf = SVM(cantidad_de_iteraciones=1000)
clf.entrenamiento(X_train, y_train)
predictions = clf.predecir(X_test)

# Efectuamos la prediccion para el conjunto de datos de entrada seleccionado
def calcular_tasa_de_acierto(y_true, y_pred):
    """calcula la tasa de acierto del modelo"""
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

print("SVM Accuracy: ", calcular_tasa_de_acierto(y_test, predictions))

from sklearn.metrics import confusion_matrix

# Definimos clases
class_labels = ['not fire', 'fire']

# Predecimos
predictions = clf.predecir(X_test)

# Calculamos accuracy
accuracy = calcular_tasa_de_acierto(y_test, predictions)

# Chequeamos los valores unicos de y_test
unique_labels = np.unique(y_test)
print("Etiqueras de y_test:", unique_labels)

# Actualizamos si es necesario
if not set(class_labels).issubset(unique_labels):
    class_labels = unique_labels

# Mostramos matriz de confusion
cm = confusion_matrix(y_test, predictions, labels=class_labels)
print("Confusion Matrix:")
print(cm)
