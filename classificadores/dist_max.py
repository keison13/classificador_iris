import numpy as np

class MaxDistance:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean_vectors = {}
        for cls in self.classes:
            self.mean_vectors[cls] = np.mean(X[y == cls], axis=0)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Calcula a função de decisão para cada classe
        scores = {
            cls: self.funcao_decisao(x, mean_vec)
            for cls, mean_vec in self.mean_vectors.items()
        }
        # Retorna a classe com o maior valor de função de decisão
        return max(scores, key=scores.get)

    def funcao_decisao(self, vetor_caracteristica, vetor_media_classe):
        return np.dot(np.transpose(vetor_media_classe), vetor_caracteristica) - np.dot(0.5 * np.transpose(vetor_media_classe), vetor_media_classe)
    
