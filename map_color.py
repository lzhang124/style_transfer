import numpy as np

try:
    reduce
except NameError:
    from functools import reduce

def _matmul(*args):
    return reduce(np.matmul, args)

def map_color(content, style, mapping):
    if mapping not in globals() or mapping == 'map_color':
        raise ValueError("Color mapping algorithm not implemented.")
    return globals()[mapping](content, style)

def linear(content, style):
    flat_c = content.reshape(-1,3)
    flat_s = style.reshape(-1,3)
    mu_c = np.mean(flat_c, axis=0)
    mu_s = np.mean(flat_s, axis=0)
    cov_c = np.cov(flat_c, rowvar=False)
    cov_s = np.cov(flat_s, rowvar=False)

    v_c, U_c = np.linalg.eig(cov_c)
    v_s, U_s = np.linalg.eig(cov_s)
    V_c = np.diag(np.sqrt(v_c))
    V_s_inv = np.diag(1. / np.sqrt(v_s))
    
    A = _matmul(U_c, V_c, U_c.T, U_s, V_s_inv, U_s.T)
    b = mu_c - _matmul(A, mu_s)

    h, w, _ = style.shape
    mapped = np.zeros(style.shape)
    for i in range(h):
        for j in range(w):
            mapped[i][j] = _matmul(A, style[i][j]) + b

    return mapped
