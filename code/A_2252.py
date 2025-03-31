import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# Parameters
L = 40.0
Nx = 512
dt = 0.01
T_total = 20.0

x = np.linspace(-L/2, L/2, Nx, endpoint=False)
dx = x[1] - x[0]
k = 2 * np.pi * fftfreq(Nx, d=dx)

# Initial wave function
psi_initial = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2).astype(np.complex128)
norm = np.sqrt(np.sum(np.abs(psi_initial)**2 * dx))
psi = psi_initial / norm

linear_phase = np.exp(-0.5j * (k**2) * dt)

Nt = int(T_total / dt)
t_values = np.linspace(0, T_total, Nt+1)
rho = np.zeros((Nt+1, Nx))
w = np.zeros(Nt+1)
rho[0] = np.abs(psi)**2
w[0] = np.sum(x**2 * rho[0]) * dx

for n in range(Nt):
    # Potential and Nonlinear terms
    potential = 0.5 * x**2
    nonlinear = 0.5 * np.abs(psi)**2
    psi *= np.exp(-1j * (potential + nonlinear) * dt / 2)

    # Kinetic term (Fourier space)
    psi_k = fft(psi)
    psi_k *= linear_phase
    psi = ifft(psi_k)

    # Potential and Nonlinear terms (second half-step)
    potential = 0.5 * x**2
    nonlinear = 0.5 * np.abs(psi)**2
    psi *= np.exp(-1j * (potential + nonlinear) * dt / 2)

    # Store results
    rho[n+1] = np.abs(psi)**2
    w[n+1] = np.sum(x**2 * rho[n+1]) * dx

# Plot density evolution heatmap
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_values, x, rho.T, shading='auto', cmap='viridis')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.ylim(-4, 4)
plt.title('Density Evolution $\\rho(x,t)$')
plt.colorbar(label='$|\\psi|^2$')
plt.savefig("./figure/density_evolution_with_x_from_{x[0]}_to_{x[-1]}_2252.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot wave packet width evolution
plt.figure(figsize=(10, 6))
plt.plot(t_values, w)
plt.xlabel('Time (t)',fontsize=20)  
plt.ylabel('Wave Packet Width $\\langle x^2 \\rangle$',fontsize=20)
plt.title('Evolution of Wave Packet Width',fontsize=20)
plt.grid(True)
plt.savefig("./figure/wave_packet_width_evolution_2252.png", dpi=300, bbox_inches='tight')
plt.show()