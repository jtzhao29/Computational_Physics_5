import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 3
N = 512
T = 20
dt = 0.001
x = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# Initial wave function
psi = (1 / np.sqrt(np.sqrt(2 * np.pi))) * np.exp(-x**2 / 2)

# Potential and nonlinear term
V = 0.5 * x**2

# Time evolution
timesteps = int(T / dt)
rho = np.zeros((timesteps, N))

for n in range(timesteps):
    # Store density
    rho[n, :] = np.abs(psi)**2

    # Step 1: Nonlinear term
    psi = psi * np.exp(-1j * 0.5 * dt * np.abs(psi)**2/2)* np.exp(-1j * 0.5 * dt * V)

    # Step 2: Kinetic term (Fourier space)
    psi_hat = np.fft.fft(psi)
    psi_hat = psi_hat * np.exp(-1j * 0.5 * dt * k**2)
    psi = np.fft.ifft(psi_hat)

    # Step 3: Potential term
    psi = psi * np.exp(-1j * 0.5 * dt * V)* np.exp(-1j * 0.5 * dt * np.abs(psi)**2/2)
    # Plot heatmap of density

t = np.linspace(0, T, timesteps)
X, T_grid = np.meshgrid(x, t)

plt.figure(figsize=(8, 6))
plt.pcolormesh(T_grid, X, rho, shading='auto', cmap='viridis')
plt.colorbar(label=r"$\rho(x, t)$")
plt.xlabel(" $t$", fontsize=20)
plt.ylabel(" $x$", fontsize=20)
plt.title(r"Density $\rho(x, t)$ vs time" , fontsize=20)

path= f"./figure/density_evolution_with_x_from_{x[0]}_to_{x[-1]}.png"
plt.savefig(path, dpi=300, bbox_inches='tight')
plt.show()

# Compute wave packet width w(t)
w_t = np.zeros(timesteps)

for n in range(timesteps):
    # Calculate x^2 <|ψ|^2> -> wave packet width
    w_t[n] = np.sum(x**2 * rho[n, :]) * dx

    # Step 1: Nonlinear term
    psi = psi * np.exp(-1j * 0.5 * dt * np.abs(psi)**2/2)* np.exp(-1j * 0.5 * dt * V)
    # Step 2: Kinetic term (Fourier space)
    psi_hat = np.fft.fft(psi)
    psi_hat = psi_hat * np.exp(-1j * 0.5 * dt * k**2)
    psi = np.fft.ifft(psi_hat)

    # Step 3: Potential term
    psi = psi * np.exp(-1j * 0.5 * dt * V)* np.exp(-1j * 0.5 * dt * np.abs(psi)**2/2)

# Plot wave packet width w(t)
plt.figure(figsize=(8, 6))
plt.plot(t, w_t, label=r"$w(t) = \langle x^2 \rangle(t)$")
plt.xlabel("Time $t$")
plt.ylabel("Wave Packet Width $w(t)$")
plt.title("Evolution of Wave Packet Width")
plt.legend()
plt.grid()
path=r"./figure/Wave Packet Width $w(t)$"
plt.savefig(path, dpi=300, bbox_inches='tight')
plt.show()