import numpy as np

class BayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.covs = {}
        self.priors = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.covs[cls] = np.cov(X_cls, rowvar=False)
            self.priors[cls] = len(X_cls) / len(X)

    def discriminant_function(self, x, cls):
        
        """ d_i(x) = ln P(w_i) - 0.5 * ln|C_i| - 0.5 * (x - μ_i)^T C_i⁻¹ (x - μ_i) """
        
        mean = self.means[cls]
        cov = self.covs[cls]
        prior = self.priors[cls]

        # Termo 1: ln P(w_i)
        term1 = np.log(prior)

        # Termo 2: -0.5 * ln|C_i|
        cov_det = np.linalg.det(cov)
        term2 = -0.5 * np.log(cov_det)

        # Termo 3: -0.5 * (x - μ_i)^T C_i⁻¹ (x - μ_i)
        diff = x - mean
        cov_inv = np.linalg.inv(cov)
        term3 = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)

        return term1 + term2 + term3

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])

    def _predict_sample(self, x):
        discriminants = {}
        for cls in self.classes:
            discriminants[cls] = self.discriminant_function(x, cls)

        # Retorna a classe com o maior valor discriminante
        return max(discriminants, key=discriminants.get)

