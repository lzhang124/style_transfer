import numpy as np
from skimage import color
from sklearn.decomposition import PCA

try:
    reduce
except NameError:
    from functools import reduce

def matmul(*args):
    return reduce(np.matmul, args)

def identity(arg):
    return arg

def rgb2color(colorspace, *images):
    f = identity if colorspace == 'rgb' else getattr(color, 'rgb2{}'.format(colorspace))
    for image in images:
        yield f(np.clip(image.astype(np.float), 0, 255) / 255)

def color2rgb(colorspace, *images):
    f = identity if colorspace == 'rgb' else getattr(color, '{}2rgb'.format(colorspace))
    for image in images:
        yield f(np.clip(image.astype(np.float), 0, 255)) * 255

def rand_rot_matrix():
    theta, phi, z = np.random.uniform(size=(3,))
    theta = theta * 2.0 * np.pi
    phi = phi * 2.0 * np.pi
    z = z * 2.0
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    return (np.outer(V, V) - np.eye(3)).dot(R)

def map_color(content, style, mapping):
    colorspace, alg = mapping.split('-')
    content, style = rgb2color(colorspace, content, style)
    mapped = globals()[alg](content, style)
    mapped, = color2rgb(colorspace, mapped)
    return mapped

def lin(content, style):
    flat_c = content.reshape(-1,3)
    flat_s = style.reshape(-1,3)
    mu_c = np.mean(flat_c, axis=0)
    mu_s = np.mean(flat_s, axis=0)
    std_c = np.std(flat_c, axis=0)
    std_s = np.std(flat_s, axis=0)

    mapped = (std_c / std_s) * (style - mu_s) + mu_c
    return mapped

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
    return mapped

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
    return mapped

def pca(content, style):
    flat_c = content.reshape(-1,3)
    flat_s = style.reshape(-1,3)
    pca_c = PCA(n_components=3)
    pca_s = PCA(n_components=3)
    pca_c.fit(flat_c)
    pca_s.fit(flat_s)

    mapped = pca_c.inverse_transform(pca_s.transform(flat_s)).reshape(style.shape)
    return np.clip(mapped, 0, 255)

def rot(content, style, iters=10):
    flat_c = content.reshape(-1,3)
    flat_s = style.reshape(-1,3)
    mapped = flat_s.copy()

    for i in range(iters):
        rot_matrix = rand_rot_matrix()
        inv_matrix = np.linalg.inv(rot_matrix)
        rot_c = matmul(flat_c, rot_matrix)
        mapped = matmul(mapped, rot_matrix)
        mapped = lin(rot_c, mapped)
        mapped = matmul(mapped, inv_matrix)
    
    mapped = mapped.reshape(style.shape)
    return np.clip(mapped, 0, 255)
