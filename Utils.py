import numpy as np

def Normalized_cross_corr(a,b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a.transpose(), b.transpose(), 'full')
    return c