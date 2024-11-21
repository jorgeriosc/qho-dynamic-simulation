import math
import numpy as np


def lapl_cart(mat, dx, dy):
    """
    Compute the laplacian using `numpy.gradient` twice.
    """
    grad_y, grad_x = np.gradient(mat, dy, dx)
    grad_xx = np.gradient(grad_x, dx, axis=1)
    grad_yy = np.gradient(grad_y, dy, axis=0)
    return (grad_xx, grad_yy)


def lapl_pol(f: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the laplacian in polar coordinates using numpy.gradient twice.
    theta corresponds to axis=0
    r corresponds to axis=1
    """

    grad_th, grad_r = np.gradient(f, theta, r)
    grad_rr = np.gradient(grad_r, r, axis=1)
    grad_thth = np.gradient(grad_th, theta, axis=0)
    
    return grad_rr + 1 / r * grad_r + 1 / r**2 * grad_thth


def abs2(x):
    return x.real**2 + x.imag**2


def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return (x, y)


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return (x, y, z)


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)

    return (r, theta, phi)


def hammer_ang2cart(theta: np.ndarray, phi: np.ndarray) -> tuple:
    x = math.sqrt(8) * np.sin(theta) * np.sin(phi / 2) / np.sqrt(1 + np.sin(theta) * np.cos(phi / 2))
    y = math.sqrt(2) * np.cos(theta) / np.sqrt(1 + np.sin(theta) * np.cos(phi / 2))

    return (x, y)


# def explicit_hermite(n, x):
#     # Hermite polynomials explicit representation is faster than SciPy's algorithm
#     sum = 0
    
#     for k in range(n//2 + 1):
#         sum += (-1)**k / (math.factorial(k) * math.factorial(n - 2*k)) * (2*x)**(n - 2*k)
    
#     return math.factorial(n) * sum


# def scipy_genlaguerre(n, alpha, x):
#     # SciPy's generalized Laguerre algorithm is faster than the explicit representation
#     L = special.genlaguerre(n, alpha)
    
#     return L(x)
