import numpy as np

class MinDistance:

    def fit(self, X, y):
        """
        Aprende os protótipos (médias) de cada classe.
        
        Parâmetros:
            X: array (n_amostras, n_features)
            y: array (n_amostras,)
        """
        self.classes = np.unique(y)
        self.prototypes = {}
        for c in self.classes:
            X_c = X[y == c]
            self.prototypes[c] = np.mean(X_c, axis=0)

    def predict(self, X):
        """
        Classifica cada amostra pela distância ao protótipo mais próximo.
        
        Parâmetros:
            X: array (n_amostras, n_features)
        
        Retorna:
            Array de rótulos previstos.
        """
        preds = []
        for x in X:
            distances = {c: np.linalg.norm(x - m) for c, m in self.prototypes.items()}
            pred = min(distances, key=distances.get)
            preds.append(pred)
        return np.array(preds)

    def print_parameters(self):

        print("\n--- PARÂMETROS APRENDIDOS ---")
        for cls in self.classes:
            m_i = self.prototypes[cls]
            m_i_str = np.array2string(m_i, precision=4, floatmode='fixed')
            print(f"Classe {cls} -> Protótipo m_i = {m_i_str}")