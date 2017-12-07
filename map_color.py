import numpy as np
from PIL import Image
from skimage import color

try:
    reduce
except NameError:
    from functools import reduce

def matmul(*args):
    return reduce(np.matmul, args)

def rgb2color(colorspace, *images):
    f = getattr(color, 'rgb2{}'.format(colorspace))
    for image in images:
        yield f(image.astype(np.float) / 255)

def color2rgb(colorspace, *images):
    f = getattr(color, '{}2rgb'.format(colorspace))
    for image in images:
        yield f(image.astype(np.float)) * 255

def map_color(content, style, mapping):
    colorspace, alg = mapping.split('-')
    if colorspace != 'rgb':
        content, style = rgb2color(colorspace, content, style)
    mapped = globals()[alg](content, style)
    if colorspace != 'rgb':
        mapped, = color2rgb(colorspace, mapped)
    return mapped

def lin(content, style):
    flat_c = content.reshape(-1,3)
    flat_s = style.reshape(-1,3)
    mu_c = np.mean(flat_c, axis=0)
    mu_s = np.mean(flat_s, axis=0)
    std_c = np.std(flat_c, axis=0)
    std_s = np.std(flat_s, axis=0)

    h, w, _ = style.shape
    mapped = np.zeros(style.shape)
    for i in range(h):
        for j in range(w):
            mapped[i][j] = (std_c / std_s) * (style[i][j] - mu_s) + mu_c
    return np.clip(mapped, 0, 255)

def eig(content, style):
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
    
    A = matmul(U_c, V_c, U_c.T, U_s, V_s_inv, U_s.T)
    b = mu_c - matmul(A, mu_s)

    h, w, _ = style.shape
    mapped = np.zeros(style.shape)
    for i in range(h):
        for j in range(w):
            mapped[i][j] = matmul(A, style[i][j]) + b
    return np.clip(mapped, 0, 255)

def chol(content, style):
    flat_c = content.reshape(-1,3)
    flat_s = style.reshape(-1,3)
    mu_c = np.mean(flat_c, axis=0)
    mu_s = np.mean(flat_s, axis=0)
    cov_c = np.cov(flat_c, rowvar=False)
    cov_s = np.cov(flat_s, rowvar=False)

    L_c = np.linalg.cholesky(cov_c)
    L_s_inv = np.linalg.inv(np.linalg.cholesky(cov_s))
        
    A = matmul(L_c, L_s_inv)
    b = mu_c - matmul(A, mu_s)

    h, w, _ = style.shape
    mapped = np.zeros(style.shape)
    for i in range(h):
        for j in range(w):
            mapped[i][j] = matmul(A, style[i][j]) + b
    return np.clip(mapped, 0, 255)
