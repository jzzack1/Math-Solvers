import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Grid parameters
N = 10
L = 0.1
dx = L / (N + 1)
x = np.linspace(dx, L - dx, N)
x = x -L/2

# Physical constants
m = 1
hbar = 1
omega = 1


# Kinetic energy operator
main = np.diag(np.full(N, -2))
upper = np.diag(np.ones(N-1), 1)
lower = np.diag(np.ones(N-1), -1)
laplacian = main + upper + lower
KE = -(hbar**2) / (2*m*dx**2) * laplacian

# Potential energy operator (harmonic oscillator)
Vx = 0.5 * m * omega**2 * x**2
V = np.diag(Vx)

# Hamiltonian and solution
H = KE + V
E, psi = eigh(H)

print("Lowest energy eigenvalues:")
print(E[:5])


#Plot of Eigenfunctions
psi0 = psi[:, 0]
psi0 = psi0 / np.sqrt(np.sum(psi0**2) * dx)
plt.plot(x, psi0**2)
plt.title("Schrodinger numerical solution")
plt.xlabel("x")
plt.ylabel("psi")
plt.grid(True)
plt.show()








