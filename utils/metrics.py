import numpy as np

# Metricas calculadas
def calcular_metricas(cm):
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    po = (tp + tn) / total
    pe = ((tp + fp)*(tp + fn) + (fn + tn)*(fp + tn)) / (total ** 2)
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # b = 1
    f2 = 3 * (precision * recall) / ((4 * precision) + recall) if ((4 * precision) + recall) > 0 else 0 # b = 2
    f3 = 1.5 * (precision * recall) / ((0.25 * precision) + recall) if ((0.25 * precision) + recall) > 0 else 0 # b = 0.5
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    mcc = (((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))) 
    return acc, precision, recall, specificity, f1, f2, f3, kappa, mcc