import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PerceptronDelta(BaseEstimator, ClassifierMixin):
    
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        """
        Inicializa o Perceptron com Regra Delta.
        
        Parâmetros:
        -----------
        eta : float
            Taxa de aprendizado (entre 0.0 e 1.0)
        n_iter : int
            Número de iterações sobre o dataset de treino
        random_state : int
            Semente aleatória para inicialização dos pesos
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.errors_ = []
        self.w_ = None
        
    def fit(self, X, y):
        """
        Treina o modelo usando a regra Delta.
        
        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados de treinamento
        y : array-like, shape = [n_samples]
            Valores alvo (rótulos das classes)
            
        Retorna:
        --------
        self : objeto
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # Lista para armazenar o erro em cada época
        self.errors_ = []
        
        # Treinamento para cada época
        for _ in range(self.n_iter):
            errors = 0
            mse = 0.0  # Erro quadrático médio
            
            # Para cada amostra no conjunto de treinamento
            for xi, target in zip(X, y):
                # Calcular a saída do modelo atual
                net_input = self.net_input(xi)
                output = self.activation(net_input)
                
                # Calcular o erro
                error = target - output
                
                # Atualizar os pesos usando a regra Delta
                self.w_[1:] += self.eta * error * xi
                self.w_[0] += self.eta * error
                
                # Acumular o erro para esta amostra
                errors += int(error != 0.0)
                mse += error**2
            
            # Adicionar o erro médio desta época à lista
            self.errors_.append(mse / len(y))
        
        return self
    
    def net_input(self, X):
        """
        Calcula a entrada da rede (produto escalar dos pesos e entradas + bias).
        
        Parâmetros:
        -----------
        X : array-like
            Vetor de características de uma amostra
            
        Retorna:
        --------
        float : entrada da rede
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """
        Aplica a função de ativação (limiar) à entrada da rede.
        
        Parâmetros:
        -----------
        X : array-like
            Entrada da rede
            
        Retorna:
        --------
        int : classe predita (0 ou 1)
        """
        return np.where(X >= 0.0, 1, 0)
    
    def predict(self, X):
        """
        Realiza a predição usando o modelo treinado.
        
        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados para predição
            
        Retorna:
        --------
        array : classes preditas
        """
        return self.activation(self.net_input(X))