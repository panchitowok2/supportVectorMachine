import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)

    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b 
        return self.cls_map[idx] * linear_model >= 1
    
    def _get_gradients(self, constrain, x, idx):
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db
    
    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db
    
    def fit(self, X, y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx)
                dw, db = self._get_gradients(constrain, x, idx)
                self._update_weights_bias(dw, db)
    
    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)

# Cargamos el conjunto de datos de incendio
data =  pd.read_csv('dataset.csv', sep=",")
print(data)

# Reemplazar los valores del campo categorico a numerico para el algoritmo
# Reemplazar los valores utilizando el método replace
data['Classes'] = data['Classes'].replace({'fire': 1, 'not fire': 0})
print('El data set despues de los cambios: ')
print(data)

# distributing the dataset into two components X and Y
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# Splitting the X and Y into the
# Training set and Testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


clf = SVM(n_iters=1000)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

print("SVM Accuracy: ", accuracy(y_test, predictions))

# Definir los colores para cada clase
colores = {0: 'red', 1: 'blue'}

# Definir los nombres de las clases
nombres_clases = {0: 'Clase not fire', 1: 'Clase fire'}

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