import random
import math
import numpy as np

# Função de ativação sigmoid: para o intervalo (0, 1)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivada da função sigmoid
def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1 - s)

# Codificação one-hot para 3 classes 
def one_hot(label):
    if label == 0: return [1, 0, 0]
    if label == 1: return [0, 1, 0]
    return [0, 0, 1]

class RedeNeuralRasa:
    def __init__(self):
        self.rede = None
        self.errors_ = []

    # Inicializa os pesos e biases aleatoriamente
    
    def inicializar(self, entradas, ocultas, saidas):
        # Estrutura da rede:

        self.rede = {
            "w1": [[random.uniform(-1, 1) for _ in range(entradas)] for _ in range(ocultas)],  # pesos da entrada para a camada oculta
            "b1": [random.uniform(-1, 1) for _ in range(ocultas)],                             # bias da camada oculta
            "w2": [[random.uniform(-1, 1) for _ in range(ocultas)] for _ in range(saidas)],   # pesos da camada oculta para a saída
            "b2": [random.uniform(-1, 1) for _ in range(saidas)]                              # bias da camada de saída
        }

    # Propagação direta (feedforward)
    def feedforward(self, entrada):
        z1 = []  # Potencial de ativação da camada oculta (pré-ativação)
        a1 = []  # Saída da camada oculta (pós-ativação)

        # Camada oculta: soma ponderada + ativação
        for i in range(len(self.rede["w1"])):
            soma = sum(entrada[j] * self.rede["w1"][i][j] for j in range(len(entrada))) + self.rede["b1"][i]
            z1.append(soma)
            a1.append(sigmoid(soma))

        z2 = []  # Potencial da camada de saída
        a2 = []  # Saída da rede

        # Camada de saída: soma ponderada + ativação
        for i in range(len(self.rede["w2"])):
            soma = sum(a1[j] * self.rede["w2"][i][j] for j in range(len(a1))) + self.rede["b2"][i]
            z2.append(soma)
            a2.append(sigmoid(soma))

        return z1, a1, z2, a2

    
    def fit(self, X, y, taxa_aprendizado=0.1, epocas=1000):

        # Normalização: z-score
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
        # - entradas = número de características
        # - 6 neurônios na camada oculta
        # - 2 neurônios na camada de saída 
        
        self.inicializar(entradas=len(X[0]), ocultas=6, saidas=2)
        self.errors_ = []

        for epoca in range(epocas):
            erro_total = 0

            for entrada, esperado_index in zip(X, y):
                # Conversão para vetor [1, 0] ou [0, 1] (binário)
                esperado = [1, 0] if esperado_index == 0 else [0, 1]

                # Propagação direta
                z1, a1, z2, saida = self.feedforward(entrada)

                # Cálculo do erro na camada de saída 
                
                erro_saida = [(saida[i] - esperado[i]) * sigmoid_derivada(z2[i]) for i in range(2)]

                # Cálculo do erro na camada oculta
                
                erro_oculta = []
                for i in range(len(a1)):
                    erro = sum(erro_saida[j] * self.rede["w2"][j][i] for j in range(2)) * sigmoid_derivada(z1[i])
                    erro_oculta.append(erro)

                # Atualização dos pesos e bias da camada de saída
                for i in range(2):
                    for j in range(len(a1)):
                        self.rede["w2"][i][j] -= taxa_aprendizado * erro_saida[i] * a1[j]
                    self.rede["b2"][i] -= taxa_aprendizado * erro_saida[i]

                # Atualização dos pesos e bias da camada oculta
                for i in range(len(self.rede["w1"])):
                    for j in range(len(self.rede["w1"][i])):
                        self.rede["w1"][i][j] -= taxa_aprendizado * erro_oculta[i] * entrada[j]
                    self.rede["b1"][i] -= taxa_aprendizado * erro_oculta[i]

                # Soma do erro quadrático da época 
                erro_total += sum((saida[i] - esperado[i]) ** 2 for i in range(2))

            # Armazena o erro quadratico medio de cada época para visualização futura 
            self.errors_.append(erro_total/len(X))

    # Predição: retorna o índice da saída com maior valor
    def predict(self, X):
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        resultados = []
        for entrada in X:
            _, _, _, saida = self.feedforward(entrada)
            resultados.append(np.argmax(saida)) 
        return np.array(resultados)