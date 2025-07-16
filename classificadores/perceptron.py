import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.use_quadratic = False
        self.errors_ = []  # Vai armazenar o erro por época

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        self.errors_ = []  # Limpa erros para armazenar o histórico

        # Fase Perceptron clássico (atualização por regra do perceptron)
        for _ in range(self.n_iter):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                prediction = 1 if linear_output >= 0 else 0
                error = y[idx] - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error
                    errors += 1
            self.errors_.append(errors)

            if errors == 0:
                break

        # Se convergiu perfeitamente, para aqui
        if errors == 0:
            return
        
        # Fase de ajuste quadrático (gradiente descendente)
        self.use_quadratic = True
        quadratic_errors = []  # Para guardar MSE por época nessa fase
        for _ in range(self.n_iter):
            gradient = np.zeros(self.n_features)
            bias_gradient = 0
            for idx, x_i in enumerate(X):
                error = y[idx] - (np.dot(x_i, self.weights) + self.bias)
                gradient += error * x_i
                bias_gradient += error
            self.weights += self.learning_rate * gradient
            self.bias += self.learning_rate * bias_gradient

            # Calcula MSE dessa época e armazena
            y_pred = self.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            quadratic_errors.append(mse)

        # Para diferenciar dos erros da fase anterior, você pode guardar em outro atributo:
        self.quadratic_errors_ = quadratic_errors

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
