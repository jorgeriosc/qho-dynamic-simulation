import math
import numpy as np
from scipy import special


def eigen1D(s, n, x):
    psi = ( (2 / (math.pi * s**2))**(1/4) / math.sqrt(2**n * math.factorial(n))
          * np.exp(-x**2 / s**2)
          * special.eval_hermite(n, math.sqrt(2) / s * x) )
    
    return psi


def eigen2D_cart(s, nx, ny, x, y):
    psi = ( 1 / (s * math.sqrt(2**(nx + ny - 1) * math.pi * math.factorial(nx) * math.factorial(ny)))
          * np.exp(-(x**2 + y**2) / s**2)
          * special.eval_hermite(nx, math.sqrt(2) / s * x)
          * special.eval_hermite(ny, math.sqrt(2) / s * y) )

    return psi


def eigen3D_cart(s, nx, ny, nz, x, y, z):
    psi = ( (2 / (math.pi * s**2))**(3/4)
          / math.sqrt(2**(nx + ny + nz) * math.factorial(nx) * math.factorial(ny) * math.factorial(nz))
          * np.exp(-(x**2 + y**2 + z**2) / s**2)
          * special.eval_hermite(nx, math.sqrt(2) / s * x)
          * special.eval_hermite(ny, math.sqrt(2) / s * y)
          * special.eval_hermite(nz, math.sqrt(2) / s * z) )

    return psi


def eigen2D_pol(s, n, m, r, theta):
    psi = ( math.sqrt(2 * math.factorial(n) / (math.pi * math.factorial(n + abs(m)) * s**2))
          * (math.sqrt(2) / s * r)**abs(m)
          * np.exp(-r**2 / s**2)
          * special.eval_genlaguerre(n, abs(m), 2 * r**2 / s**2)
          * np.exp(1j * m * theta) )

    return psi


def eigen3D_sph(s, n, l, m, r, theta, phi):
    psi = ( math.sqrt(2**(l + 5/2) * math.factorial(n) / (s**3 * special.gamma(n + l + 3/2)))
          * (r / s)**l
          * np.exp(- r**2 / s**2)
          * special.eval_genlaguerre(n, l + 1/2, 2 * r**2 / s**2)
          * sph_harm(l, m, theta, phi) )

    return psi


# Faster than SciPy's special.sph_harm() 
def sph_harm(l, m, theta, phi):
    Y = ( math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
        * special.lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi) )

    return Y
