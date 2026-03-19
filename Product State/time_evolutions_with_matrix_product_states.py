#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:06:45 2025

@author: rosatrullcastanyer
"""
import time
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# Configuración para usar LaTeX
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# --- PARAMETERS --------------------------------------------------------------
L = 50               # system size
Jxy = 1.0            # XY coupling
Jz = 2.8             # Z coupling

d = 2                # physical dimension
dt = 0.1             # time step

# time grid: 0:dt:L (inclusive)
time_steps = np.arange(0.0, L + dt, dt)
n_steps = len(time_steps)

chi_max = 15         # max bond dimension
w = 0.0              # discarded weight accumulator

# --- PAULI MATRICES ----------------------------------------------------------
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

Sp = Sx + 1j * Sy    # raising operator
Sm = Sx - 1j * Sy    # lowering operator

# --- HAMILTONIAN FOR THE HEISSENBERG MODEL XXZ -------------------------------
H = 0.5 * Jxy * (np.kron(Sp, Sm) + np.kron(Sm, Sp)) + Jz * np.kron(Sz, Sz)
U = expm(-1j * dt * H)
U_half = expm(-1j * dt/2 * H)

# --- INITIAL STATE MPS -------------------------------------------------------
lambda_list = [np.array([1], dtype=complex) for _ in range(L+1)]
Gamma = [np.zeros((1,1,d), dtype=complex) for _ in range(L)]

spin_up_1 = [25,26]
spin_up = [s - 1 for s in spin_up_1] # since pyhton indices start with 0
for n in range(L):
    if n in spin_up:
        Gamma[n][0, 0, 0] = 1.0    # spin up
    else:
        Gamma[n][0, 0, 1] = 1.0    # spin down

# Check
spins = []
for n in range(L):
    A = Gamma[n]                  # per accedir al Gamma de la posició n
    if A[0, 0, 0] == 1:
        spins.append("↑")
    elif A[0, 0, 1] == 1:
        spins.append("↓")

print("Initial state:", "".join(spins))

# ------- FUNCTIONS ----------
def updateBond(Gamma, lambda_list, j, u, chi, w):
    """
    Update bond between sites j and j+1
    Gamma: list of Gamma tensors
    Lambda: list of lambda vectors
    u: two-site gate (4x4)
    chi: max bond dimension
    w_sum: cumulative sum of discarded weights (for normalization monitoring)
    returns (Gamma, Lambda, w_sum)
    """
    # get dimensions
    Dl, Dm, d = Gamma[j].shape
    _, Dr, _ = Gamma[j + 1].shape
    """
    Dl: left bond dimension of site j.
    Dm: bond dimension connecting sites j and j+1.
    Dr: right bond dimension of site j+1.
    d: physical dimension per site.
    """

    lambdaL = lambda_list[j].astype(complex)      # left of site j (since pyhton indeces start with 0)
    lambdaM = lambda_list[j + 1].astype(complex)  # between j and j+1
    lambdaR = lambda_list[j + 2].astype(complex)  # right of site j+1
    # λ[j] -- Γ[j] -- λ[j+1] -- Γ[j+1] -- λ[j+2]
    # Schmidt coefficients on the bonds connecting site j-1 ↔ j, j ↔ j+1, and j+1 ↔ j+2.
    
    #Build the two-site Θ tensor formed by contracting the two neighboring MPS tensors with their adjacent λ (Schmidt) coefficients
    # M: Dl x Dm x d
    M = lambdaL[:, None, None] * Gamma[j] * lambdaM[None, :, None]
    # N: Dm x Dr x d
    N = Gamma[j + 1] * lambdaR[None, :, None]
    
    # The object Θ is the two-site tensor formed by contracting everything around sites j and j+1
    # Build theta(left,right,s1,s2) = sum_middle Mj(left,middle,s1)*Nj(middle,right,s2)
    # resulting theta shape: Dl x Dr x d x d
    theta = np.zeros((Dl, Dr, d, d), dtype=complex)
    for s1 in range(d):
        for s2 in range(d):
            # M[:,:,s1] is (Dl x Dm); N[:,:,s2] is (Dm x Dr) (: means take all elements along this axis)
            theta[:, :, s1, s2] = M[:, :, s1] @ N[:, :, s2]

    # Apply gate: reshape and multiply
    theta = theta.reshape((Dl * Dr, d * d))        # (Dl*Dr) x (d*d) (taht way we can multiply as matrices)
    theta = theta @ u                              # (Dl*Dr) x (d*d)
    theta = theta.reshape((Dl, Dr, d, d))
    # permute to (Dl, d, Dr, d) -> then reshape to (Dl*d, Dr*d)
    theta = np.transpose(theta, (0, 2, 1, 3)).reshape((Dl * d, Dr * d))

    # SVD
    U, s_vals, Vdagger = np.linalg.svd(theta, full_matrices=False)
    # s_vals is 1D array of singular values

    # Truncation
    Dm_new = s_vals.shape[0]
    norm2 = np.sum(s_vals**2)
    if Dm_new > chi:
        # keep only the top chi singular values
        lambda_keep = s_vals[:chi].copy()
        U = U[:, :chi].copy()
        Vdagger = Vdagger[:chi, :].copy()
        Dm_new = chi

        # discarded weight using \omega = 1 - \sum (\lambda_i^2 of kept)
        omega = 1.0 - np.sum(lambda_keep**2) / norm2 # we divide by the norm to make sure it's normalize since numerical code can provoque rounding errors
        w += float(omega)

        # normalize the kept singular values
        lambda_keep = lambda_keep / np.sqrt(1.0 - omega)

    else:
        # no truncation \omega = 0
        omega = 0.0
        lambda_keep = s_vals / np.sqrt(1.0 - omega)

    # Store Lambda (as vector)
    lambda_list[j + 1] = lambda_keep.real.copy()  # bond between j and j+1

    # Reshape U and Vdagger to reconstruct Gamma tensors
    # U shape: (Dl*d, Dm_new) -> reshape (Dl, d, Dm_new)
    U = U.reshape((Dl, d, Dm_new))
    # want Gamma[j] shaped (Dl, Dm_new, d): permute (0,2,1)
    # avoid dividing by extremely small λ entries
    epsilon = 1e-15
    lambdaL_safe = lambdaL.copy()
    lambdaL_safe[np.abs(lambdaL_safe) < epsilon] = 1.0
    lambdaR_safe = lambdaR.copy()
    lambdaR_safe[np.abs(lambdaR_safe) < epsilon] = 1.0

    Gamma[j] = np.transpose(U, (0, 2, 1)) / lambdaL_safe.reshape((Dl, 1, 1))

    # Vh shape: (Dm_new, Dr*d) -> reshape (Dm_new, Dr, d)
    Vt = Vdagger.reshape((Dm_new, Dr, d))
    Gamma[j + 1] = Vt / lambdaR_safe.reshape((1, Dr, 1))

    return Gamma, lambda_list, w


def local_expectation_values(Gamma, lambda_list, Sz):
    L = len(Gamma)                                # number of sites
    expected_value = np.zeros(L, dtype=float)     # array to store ⟨Sz⟩ at each site
    for j in range(L):
        Dl, Dr, d = Gamma[j].shape
        # build local matrix M = lambda[j-1] Gamma[j] lambda[j]
        lambdaL = lambda_list[j].astype(complex).reshape((Dl, 1, 1))
        lambdaR = lambda_list[j + 1].astype(complex).reshape((1, Dr, 1))
        M = lambdaL * Gamma[j] * lambdaR          # Dl x Dr x d
        M_flat = M.reshape((Dl * Dr, d))          # (Dl*Dr) x d
        # rho_{s1,s2} = Tr(M(:,:,s1)' M(:,:,s2)) -> M_flat^dagger @ M_flat
        rho= M_flat.conj().T @ M_flat           # d x d
        expected_value[j] = np.real(np.trace(Sz @ rho))
    return expected_value


def entanglement_entropy(lambda_list):
    """
    Compute the entanglement entropy S for each bond:
    S = -sum(lambda^2 * log(lambda^2))
    """
    S = np.zeros(len(lambda_list), dtype=float)
    for j, lam in enumerate(lambda_list):
        """
        the enumerate() function is a Python built-in that lets you loop over both the index and the value of an iterable (in this case, lambda_list).
        j → is the index (an integer: 0, 1, 2, …)
        lam → is the actual element from lambda_list at that index (typically a NumPy array of Schmidt coefficients)
        """
        if lam.size > 1:  # gives the number of elements in the array lam.
            lam2 = np.real(lam)**2
            lam2 = lam2 / np.sum(lam2)  # normalize (for safety)
            nonzero = lam2[lam2 > 1e-15]
            S[j] = -np.sum(nonzero * np.log(nonzero))
        else:
            S[j] = 0.0
    return S

# -----------------------
# MAIN TIME EVOLUTION LOOP USING 2nd ORDER TROTTER SUZUKI
# -----------------------
# storage for magnetization <Sz(i,t)> over time
Sz_all = np.zeros((L, n_steps), dtype=float)

# storge for entenglement
S_all = np.zeros((L + 1, n_steps), dtype=float)

# storage for discarded weight
w_list = np.zeros(n_steps)

# percentages for progress printing (same as MATLAB style)
percentage = (np.arange(1, 11) * (n_steps // 10)).tolist()
p_index = 0
t_start = time.time()


for t_idx in range(n_steps):
    
    # 1. Half-step on even bonds (indices 1,3,5,...)
    for j in range(1, L - 1, 2):
        Gamma, lambda_list, w = updateBond(Gamma, lambda_list, j, U_half, chi_max, w)

    # 2. Full-step on odd bonds (indices 0,2,4,...)
    for j in range(0, L - 1, 2):
        Gamma, lambda_list, w = updateBond(Gamma, lambda_list, j, U, chi_max, w)

    # 3. Half-step on even bonds again (indices 1,3,5,...)
    for j in range(1, L - 1, 2):
        Gamma, lambda_list, w = updateBond(Gamma, lambda_list, j, U_half, chi_max, w)

    # Progress printing (roughly 10% increments)
    if p_index < len(percentage) and (t_idx + 1) == percentage[p_index]:
        elapsed = time.time() - t_start
        percent = 10 * (p_index + 1)
        print(f"{percent}% complete with {elapsed:.2f} seconds elapsed.")
        print(f"Weight discarded {w:g}.")
        p_index += 1

    # 4. Measure <Sz> after the full time step
    Sz_all[:, t_idx] = local_expectation_values(Gamma, lambda_list, Sz)
    S_all[:, t_idx] = entanglement_entropy(lambda_list)
    w_list[t_idx] = w

# -----------------------
# PLOTTING
# -----------------------
# --- <Sz> vs TIME-------------------------------------------------------------
# White (1,1,1) -> Blue (0,0,1)
#cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"])
cmap = LinearSegmentedColormap.from_list("blue_white_red", ["blue", "white", "red"])

plt.figure(figsize=(8, 6), dpi=500)
# Transpose Sz_all so rows = time, columns = site
plt.imshow(Sz_all.T, aspect='auto', origin='upper', extent=[1, L, time_steps[-1], time_steps[0]], cmap=cmap, vmin=-0.5, vmax=0.5)  # scale: -0.5=spin down, 0.5=spin up
plt.xlabel('Position')
plt.ylabel('Time')
plt.title(r'$\langle S_z\left(x,t\right)\rangle, J_{xy} = $' + f"{Jxy}" + r', $J_z = $' + f"{Jz}" + ', Initial product state spin up positions: ' + f"{spin_up_1}")
plt.colorbar(label=r'$\langle S_z\rangle$')
plt.show()

# --- DENSITY PROFILE FOR A SNAPCHOT ------------------------------------------
# Choose the snapshot time
t_snapshot = 20.0

# Find the index in your time_steps array closest to t_snapshot
t_idx = np.argmin(np.abs(time_steps - t_snapshot))

# Extract Sz at that time
Sz_snapshot = Sz_all[:, t_idx]

# Plot Sz vs site
plt.figure(figsize=(8,6), dpi=500)
plt.plot(np.arange(1, L+1), Sz_snapshot, linestyle='-')
plt.xlabel('Position')
plt.ylabel(r'$\langle S_z \rangle$')
plt.title(f'Snapshot of ⟨Sz⟩ at t = {time_steps[t_idx]:.2f}')
plt.grid(False)
plt.show()

# --- ENTENGLEMENT VS TIME ----------------------------------------------------
# ----
# 3D
# ----
# want axes: X = position (bond), Y = time
X, Y = np.meshgrid(np.arange(1, L + 2), time_steps)

# S_all is shaped (bond, time), but we want (time, bond)
Z = S_all.T

fig = plt.figure(figsize=(8, 6), dpi=500)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.9, linewidth=0, antialiased=True)
ax.set_xlabel('position')
ax.set_ylabel('time')
ax.set_zlabel('entanglement entropy')
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(elev=30, azim=230)
plt.tight_layout()
plt.show()
# ---
# 2D
# ---
plt.figure(figsize=(8, 6), dpi=500)
# Transpose Sz_all so rows = time, columns = site
plt.imshow(S_all.T, aspect='auto', origin='upper', extent=[1, L, time_steps[-1], time_steps[0]], cmap='inferno')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title(r'Entanglement entropy $S = -\sum_\gamma \lambda_\gamma^2 \ln \lambda_\gamma^2$')
plt.colorbar(label=r'Entropy $S$')
plt.show()

# --- DISCARDED WEIGHT VS TIME ------------------------------------------------
plt.figure(figsize=(8,6), dpi=500)
plt.plot(time_steps, w_list, linestyle='-')
plt.xlabel('Time')
plt.ylabel(r'Discarded weight $\omega$')
plt.title(r'Accumulated discarded weight vs time in the simulation')
plt.grid(False)
plt.show()