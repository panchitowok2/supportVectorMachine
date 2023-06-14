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

"""
# preprocesamiento
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# aplicamos pca
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
"""

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

"""
# Definir los colores para cada clase
colores = {-1: 'red', 1: 'blue'}

# Definir los nombres de las clases
nombres_clases = {-1: 'Clase not fire', 1: 'Clase fire'}

# Crear la figura y el subplot
fig, ax = plt.subplots()

# Iterar sobre las clases únicas en y_train
for clase in np.unique(y_train):
    # Obtener los índices de los puntos de la clase actual
    indices = np.where(y_train == clase)
    # Obtener las coordenadas x e y correspondientes a los puntos de la clase actual
    x = X_train[indices, 0]
    y = X_train[indices, 1]
    # Graficar los puntos con el color correspondiente a la clase actual
    ax.scatter(x, y, color=colores[clase], label=nombres_clases[clase])

# Agregar leyenda
ax.legend()

# Mostrar el gráfico
plt.show()

# Obtener los límites del gráfico
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

# Generar un rango de valores para x e y
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Preparar los datos para realizar las predicciones en la malla
mesh_input = np.c_[xx.ravel(), yy.ravel()]

# Realizar las predicciones en la malla
Z = clf.predecir(mesh_input)
Z = Z.reshape(xx.shape)

# Graficar el contorno de la región de decisión
plt.contourf(xx, yy, Z, alpha=0.5)

# Graficar los puntos de entrenamiento con los colores correspondientes a las clases
for clase in np.unique(y_train):
    indices = np.where(y_train == clase)
    x = X_train[indices, 0]
    y = X_train[indices, 1]
    plt.scatter(x, y, color=colores[clase], label=nombres_clases[clase])

# Agregar leyenda y título
plt.legend()
plt.title('SVM')

# Mostrar el gráfico
plt.show()
"""

from sklearn.metrics import confusion_matrix

# Define the class labels
class_labels = ['not fire', 'fire']

# Make predictions
predictions = clf.predecir(X_test)

# Calculate accuracy
accuracy = calcular_tasa_de_acierto(y_test, predictions)

# Check unique values in y_test
unique_labels = np.unique(y_test)
print("Unique labels in y_test:", unique_labels)

# Update class labels if necessary
if not set(class_labels).issubset(unique_labels):
    class_labels = unique_labels

# Create confusion matrix
cm = confusion_matrix(y_test, predictions, labels=class_labels)
print("Confusion Matrix:")
print(cm)
