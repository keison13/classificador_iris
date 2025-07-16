import numpy as np
import matplotlib.pyplot as plt

class RedeNeuralMulticlasse:
    """
    Implementação de um Perceptron Multicamadas (MLP) com uma camada oculta.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa pesos e biases da rede neural.

        Args:
            input_size (int): Nº de neurônios da entrada (features).
            hidden_size (int): Nº de neurônios da camada oculta.
            output_size (int): Nº de neurônios da camada de saída (classes).
        """
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, y_true, y_pred):
        num_samples = len(y_true)
        epsilon = 1e-9
        return - (1 / num_samples) * np.sum(y_true * np.log(y_pred + epsilon))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        y_pred = self._softmax(self.z2)
        return y_pred

    def backward(self, X, y_true, y_pred):
        num_samples = X.shape[0]

        dZ2 = y_pred - y_true
        dW2 = (1 / num_samples) * np.dot(self.a1.T, dZ2)
        db2 = (1 / num_samples) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._sigmoid_derivative(self.z1)
        dW1 = (1 / num_samples) * np.dot(X.T, dZ1)
        db1 = (1 / num_samples) * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def train(self, X_train, y_train, epochs, learning_rate):
        # Normalizar os dados
        if not hasattr(self, 'X_mean') or not hasattr(self, 'X_std'):
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0)
        X_train = (X_train - self.X_mean) / self.X_std


        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            loss = self._compute_loss(y_train, y_pred)
            self.loss_history.append(loss)

            dW1, db1, dW2, db2 = self.backward(X_train, y_train, y_pred)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Época {epoch + 1}/{epochs}, Perda (Loss): {loss:.4f}")

        return self.loss_history

    def predict(self, X):
        X = (X - self.X_mean) / self.X_std
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def obter_probabilidades(self, entrada):
        """
        Retorna as probabilidades (softmax) para uma amostra de entrada.
        
        Args:
            entrada (list ou array): Amostra com os atributos normalizados.

        Returns:
            np.array: Vetor de probabilidades para cada classe.
        """
        entrada = np.array(entrada).reshape(1, -1)
        entrada = (entrada - self.X_mean) / self.X_std
        saida = self.forward(entrada)
        return saida.flatten()
    
    def obter_arquitetura(self):
        """
        Retorna uma descrição da arquitetura da rede.
        
        Returns:
            str: Arquitetura da rede em formato descritivo.
        """
        return (
            "Arquitetura da Rede Neural:\n"
            f"- Entrada: {self.W1.shape[0]} neurônios (features)\n"
            f"- Camada Oculta: {self.W1.shape[1]} neurônios\n"
            f"- Saída: {self.W2.shape[1]} neurônios (classes)\n"
            "- Funções de ativação: Sigmoid (oculta), Softmax (saída)\n"
        )



    def prever_amostra_manual(self, entrada):
        """
        Recebe uma amostra (lista de 4 atributos), normaliza e retorna a classe prevista e as saídas da rede.
        """
        entrada = np.array(entrada).reshape(1, -1)
        entrada = (entrada - self.X_mean) / self.X_std
        saida = self.forward(entrada)
        classe = int(np.argmax(saida))
        return classe, saida.flatten()

    def plotar_erro(self):
        """
        Plota a curva de erro (loss) ao longo das épocas.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label='Erro (Loss)')
        plt.xlabel('Épocas')
        plt.ylabel('Erro')
        plt.title('Erro durante o Treinamento')
        plt.legend()
        plt.grid(True)
        plt.show()

def one_hot_encode(y, num_classes):
    """
    Codifica os rótulos em one-hot.

    Args:
        y (array): vetor de rótulos (ex: [0, 2, 1])
        num_classes (int): número total de classes

    Returns:
        np.array: matriz codificada em one-hot
    """
    return np.eye(num_classes)[y]


