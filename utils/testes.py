from scipy.stats import norm
from math import sqrt
import numpy as np

# Testes realizados
def teste_significancia(y_pred1, y_pred2, y_true):
    n = len(y_true)
    corrects1 = np.array(y_pred1 == y_true, dtype=int)
    corrects2 = np.array(y_pred2 == y_true, dtype=int)
    t1 = corrects1.mean()
    t2 = corrects2.mean()
    ag = (t1 + t2) / 2
    m = n
    c = 2
    sigma_sq = (1 / m) * (ag * (1 - ag)) / ((1 - (1 / c)) ** 2)

    if sigma_sq == 0:
        return "Variância zero — teste de significância não aplicável (classificadores idênticos ou quase)."

    z = (t1 - t2) / sqrt(sigma_sq)
    z_crit = norm.ppf(1 - 0.05 / 2)

    if abs(z) > z_crit:
        return f"Z = {z:.4f} | Z crítico = ±{z_crit:.4f} → Rejeita H0: há diferença significativa."
    else:
        return f"Z = {z:.4f} | Z crítico = ±{z_crit:.4f} → Não rejeita H0: não há diferença significativa."