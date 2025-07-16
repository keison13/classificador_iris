import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Hopfield:
    def __init__(self):
        self.W = None
        self.patterns = None
        self.pca = PCA(n_components=4)
        self.scaler = StandardScaler()

    def fit(self, X):
        # Normaliza e aplica PCA
        X_norm = self.scaler.fit_transform(X)
        X_reduced = self.pca.fit_transform(X_norm)

        # Binariza os dados
        X_binary = np.where(X_reduced > 0, 1, -1)

        # Armazena apenas o primeiro padrão (como no seu código original)
        self.patterns = X_binary[:1]

        # Treina a matriz de pesos
        self.W = self.train_hopfield(self.patterns)

    def train_hopfield(self, patterns):
        n = patterns.shape[1]
        W = np.zeros((n, n))
        for p in patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0)  # Remove auto-conexões
        return W

    def recall_async(self, pattern, steps=5):
        s = pattern.copy()
        n = len(s)
        for _ in range(steps):
            for i in range(n):
                s[i] = np.sign(np.dot(self.W[i], s))
                if s[i] == 0:
                    s[i] = 1
        return s

    def check_match(self, recovered):
        for idx, p in enumerate(self.patterns):
            if np.array_equal(recovered, p):
                return f"Padrão reconhecido: {idx}"
        return "Não corresponde a nenhum padrão armazenado (mínimo espúrio)"

    def add_noise(self, pattern, n_bits=1):
        noisy = pattern.copy()
        n = len(noisy)
        flipped = 0
        attempts = 0

        while flipped < n_bits and attempts < 10 * n_bits:
            idx = np.random.randint(0, n)
            if noisy[idx] != 0:
                noisy[idx] *= -1
                flipped += 1
            attempts += 1
        return noisy

