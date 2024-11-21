import math
import numpy as np
import qho_eigen as qho
from math_tools import *
import matplotlib.pyplot as plt


class WaveFunction:
    def __init__(self, dims: int, coord_sys: str, hbar: float, M: float, omega: float):
        self.dims = dims
        self.coord_sys = coord_sys #'cartesian','polar','spherical'

        self.hbar  = hbar
        self.M     = M
        self.omega = omega
        self.s     = math.sqrt(2 * self.hbar / (self.M * self.omega))


    def setup_coords(self, x_min: float, x_max: float, x_N: int):
        self.x = np.linspace(x_min, x_max, x_N)


    def compute_eigenfuns(self, N: int, n: np.ndarray):
        self.n = np.vstack(n, dtype=object)

        self.eigenfuns = np.zeros((N, np.size(self.x)))

        for i in range(N):
            self.eigenfuns[i] = qho.eigen1D(self.s, self.n[i][0], self.x)
    

    def init_state(self, psi0: np.ndarray):
        self.psi0 = psi0

        self.coeffs = 10 / (3 * math.sqrt(77)) * np.vstack(np.array([1, 0.7, -1.2, 2]))

        print(f'Integral of |Ψ(x,0)|² = {np.trapz(abs2(psi0), x=self.x)}')


    def time_evolve(self, t_max: float, t_N: int):
        self.t = np.linspace(0, t_max, t_N)

        self.psi = np.zeros((np.size(self.x), np.size(self.t)), dtype=np.complex128)
        self.psi[:,0] = self.psi0

        n = self.n.astype(np.float64)

        for i in range(np.size(self.t) - 1):
            self.psi[:,i+1] = np.sum(self.coeffs * self.eigenfuns * np.exp(-1j * (0.5 + n) * self.omega * self.t[i+1]), axis=0)


    def expectation_value_position(self):
        self.expect_x = np.zeros(np.size(self.t))

        for i in range(np.size(self.t)):
            self.expect_x[i] = np.trapz(self.x * abs2(self.psi[:,i]), x=self.x)



    def plot_initial_PDF(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, abs2(self.psi0), linewidth=2)
        ax.set_title('Initial Probability Density Function')
        ax.set_xlabel('Position $x$'), ax.set_ylabel(r'$|\Psi(x,0)|^2$')

        ax.set_xlim(self.x[0], self.x[-1])#, ax.set_ylim(0, 0.6)
        # ax.set_xticks(np.arange(self.x[0], self.x[-1] + 1))
        ax.grid(alpha=0.5)

        plt.show()

    
    def plot_time_evolve_PDF(self):
        fig, ax = plt.subplots()
        pcmesh = ax.pcolormesh(self.t, self.x, abs2(self.psi), cmap='inferno', vmin=0)
        ax.set_title(r'Evolution of Probability Density Function $|\Psi(x,t)|^2$')
        ax.set_xlabel('Time $t$'), ax.set_ylabel('Position $x$')

        ax.set_xlim(0, self.t[-1]), ax.set_ylim(self.x[0], self.x[-1])
        ax.set_xticks(np.linspace(0, self.t[-1], 7))
        ax.set_xticklabels(('0', r'$\pi$/3', r'2$\pi$/3', r'$\pi$', r'4$\pi$/3', r'5$\pi$/3', r'$2\pi$'))
        # ax.set_yticks(np.arange(x_min, x_max + 1))

        fig.colorbar(pcmesh, ax=ax)

        plt.show()
    

    def plot_expectation_value_position(self):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.expect_x, linewidth=2)
        ax.set_title('Expectation Value of Position')
        ax.set_xlabel('Time $t$'), ax.set_ylabel(r'Expectation Value $\langle x \rangle$')

        ax.set_xlim(0, self.t[-1]), ax.set_ylim(self.x[0], self.x[-1])
        ax.set_xticks(np.linspace(0, self.t[-1], 7))
        ax.set_xticklabels(('0', r'$\pi$/3', r'2$\pi$/3', r'$\pi$', r'4$\pi$/3', r'5$\pi$/3', r'$2\pi$'))
        # ax.set_yticks(np.arange(x_min, x_max + 1))
        ax.grid(alpha=0.5)

        plt.show()





if __name__ == '__main__':
    wavefun = WaveFunction(dims=1, coord_sys='cartesian',
                           hbar=1, M=1, omega=1)

    wavefun.setup_coords(-10.0, 10.0, 1001)

    wavefun.compute_eigenfuns(4, np.arange(4))

    wavefun.init_state(psi0=10 / (3 * math.sqrt(77))
                            * (wavefun.eigenfuns[0,:]
                            + 0.7*wavefun.eigenfuns[1,:]
                            - 1.2*wavefun.eigenfuns[2,:]
                            + 2.0*wavefun.eigenfuns[3,:]))
    
    wavefun.plot_initial_PDF()

    wavefun.time_evolve(t_max=2 * math.pi / wavefun.omega, t_N=500)
    wavefun.plot_time_evolve_PDF()

    wavefun.expectation_value_position()
    wavefun.plot_expectation_value_position()





