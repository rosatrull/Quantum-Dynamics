#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 17:15:31 2025

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
Jz = 0.0             # Z coupling

d = 2                # physical dimension
dt = 0.1             # time step

# time grid: 0:dt:L (inclusive)
time_steps = np.arange(0.0, L + dt, dt)
n_steps = len(time_steps)

chi_max = [30, 50, 100, 200]   # max bond dimension  ATENTION <S_z> and S(x,t) will be ploted only for the las chi
w = 0.0              # discarded weight accumulator

# --- PAULI MATRICES ----------------------------------------------------------
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

Sp = Sx + 1j * Sy    # raising operator
Sm = Sx - 1j * Sy    # lowering operator

# --- HAMILTONIAN FOR THE HEISSENBERG MODEL XXZ -------------------------------
def create_gates(Jxy_list, Jz_list, dt):
    """
    Creates the list of two-site time evolution operators U and U_half for each bond.
    Jxy_list/Jz_list are of size L-1 (one for each bond j, j+1).
    """
    U_list = []
    U_half_list = []
    for j in range(L - 1):
        Jxy = Jxy_list[j]
        Jz = Jz_list[j]
        
        # Hamiltonian for the bond (j, j+1)
        H = 0.5 * Jxy * (np.kron(Sp, Sm) + np.kron(Sm, Sp)) + Jz * np.kron(Sz, Sz)
        U = expm(-1j * dt * H)
        U_half = expm(-1j * dt/2 * H)
        
        U_list.append(U)
        U_half_list.append(U_half)
    return U_list, U_half_list


# --- CASE DEFINITIONS -------------------------------------------------------
num_bonds = L - 1
center_bond_idx = num_bonds // 2
center_site_idx = L // 2 - 1

cases = {
    'Case 1: XX (Jz=0, Jxy=1)': {
        'Jxy': [1.0] * num_bonds,
        'Jz': [0.0] * num_bonds,
    },
    'Case 2: XX (Jz=0, Jxy=1.0 left half, Jxy=0.5 right half)': {
        'Jxy': [1.0] * (num_bonds // 2) + [0.5] * (num_bonds - (num_bonds // 2)),
        'Jz': [0.0] * num_bonds,
    },
    'Case 3: XX with weak link (Jz=0, Jxy=1 (middle bond Jxy=0.5))': {
        # initialize all Jxy=1, later we will change the middle bond to Jxy=0.5
        'Jxy': [1.0] * num_bonds,
        'Jz': [0.0] * num_bonds,
    },
    'Case 4: XXZ (Jz=1, Jxy=1)': {
        'Jxy': [1.0] * num_bonds,
        'Jz': [1.0] * num_bonds,
    },
}
cases['Case 3: XX with weak link (Jz=0, Jxy=1 (middle bond Jxy=0.5))']['Jxy'][center_bond_idx] = 0.5

# --- FUNCTIONS --------------------------------------------------------------
def updateBond(Gamma, lambda_list, j, u_list, chi, w):
    Dl, Dm, d = Gamma[j].shape
    _, Dr, _ = Gamma[j+1].shape

    lambdaL = lambda_list[j].astype(complex)
    lambdaM = lambda_list[j+1].astype(complex)
    lambdaR = lambda_list[j+2].astype(complex)

    M = lambdaL[:, None, None] * Gamma[j] * lambdaM[None, :, None]
    N = Gamma[j+1] * lambdaR[None, :, None]

    theta = np.zeros((Dl, Dr, d, d), dtype=complex)
    for s1 in range(d):
        for s2 in range(d):
            theta[:, :, s1, s2] = M[:, :, s1] @ N[:, :, s2]

    theta = theta.reshape((Dl*Dr, d*d)) @ u_list[j]
    theta = theta.reshape((Dl, Dr, d, d))
    theta = np.transpose(theta, (0, 2, 1, 3)).reshape((Dl*d, Dr*d))

    U, s_vals, Vdagger = np.linalg.svd(theta, full_matrices=False)
    Dm_new = s_vals.shape[0]
    norm2 = np.sum(s_vals**2)
    if Dm_new > chi:
        lambda_keep = s_vals[:chi].copy()
        U = U[:, :chi].copy()
        Vdagger = Vdagger[:chi, :].copy()
        Dm_new = chi
        omega = 1.0 - np.sum(lambda_keep**2) / norm2
        w += float(omega)
        lambda_keep = lambda_keep / np.sqrt(1.0 - omega)
    else:
        omega = 0.0
        lambda_keep = s_vals / np.sqrt(1.0 - omega)

    lambda_list[j+1] = lambda_keep.real.copy()
    U = U.reshape((Dl, d, Dm_new))
    epsilon = 1e-15
    lambdaL_safe = lambdaL.copy()
    lambdaL_safe[np.abs(lambdaL_safe) < epsilon] = 1.0
    lambdaR_safe = lambdaR.copy()
    lambdaR_safe[np.abs(lambdaR_safe) < epsilon] = 1.0

    Gamma[j] = np.transpose(U, (0, 2, 1)) / lambdaL_safe.reshape((Dl, 1, 1))
    Vt = Vdagger.reshape((Dm_new, Dr, d))
    Gamma[j+1] = Vt / lambdaR_safe.reshape((1, Dr, 1))
    return Gamma, lambda_list, w

def local_expectation_values(Gamma, lambda_list, Sz):
    L = len(Gamma)
    expected_value = np.zeros(L, dtype=float)
    for j in range(L):
        Dl, Dr, d = Gamma[j].shape
        lambdaL = lambda_list[j].astype(complex).reshape((Dl,1,1))
        lambdaR = lambda_list[j+1].astype(complex).reshape((1,Dr,1))
        M = lambdaL * Gamma[j] * lambdaR
        M_flat = M.reshape((Dl*Dr, d))
        rho = M_flat.conj().T @ M_flat
        expected_value[j] = np.real(np.trace(Sz @ rho))
    return expected_value

def entanglement_entropy(lambda_list):
    S = np.zeros(len(lambda_list), dtype=float)
    for j, lam in enumerate(lambda_list):
        if lam.size > 1:
            lam2 = np.real(lam)**2
            lam2 = lam2 / np.sum(lam2)
            nonzero = lam2[lam2 > 1e-15]
            S[j] = -np.sum(nonzero * np.log(nonzero))
        else:
            S[j] = 0.0
    return S

# --- STORAGE ---------------------------------------------------------------
# Dictionary to store results: results[case_label][chi] = S_t_array
S_center_t_storage = {} 
Sz_all_storage = {}
S_all_storage = {}
w_list_storage = {}

t_start_total = time.time()

# --- MAIN LOOP OVER CASES -----------------------------------------------
for case_label, params in cases.items():
    print(f"--- Starting simulation for {case_label} ---")
    # Prepare storage for this specific case's chi comparison
    S_center_t_storage[case_label] = {}
    
    # Gates depend only on Hamiltonian, not chi, so create them once per case
    U_list, U_half_list = create_gates(params['Jxy'], params['Jz'], dt)

    for chi in chi_max:
        # Initialize MPS
        lambda_list = [np.array([1], dtype=complex) for _ in range(L+1)]
        Gamma = [np.zeros((1,1,d), dtype=complex) for _ in range(L)]
    
        spin_up_1 = list(range(1,26))
        spin_up = [s-1 for s in spin_up_1]
        for n in range(L):
            if n in spin_up:
                Gamma[n][0,0,0] = 1.0
            else:
                Gamma[n][0,0,1] = 1.0
    
        # Check
        spins = ["↑" if Gamma[n][0,0,0]==1 else "↓" for n in range(L)]
        print("Initial state:", "".join(spins))
    
        Sz_all = np.zeros((L, n_steps), dtype=float)
        S_all = np.zeros((L+1, n_steps), dtype=float)
        w_list = np.zeros(n_steps)
        w = 0.0
    
        for t_idx in range(n_steps):
            # Half-step even bonds
            for j in range(1, L-1, 2):
                Gamma, lambda_list, w = updateBond(Gamma, lambda_list, j, U_half_list, chi, w)
            # Full-step odd bonds
            for j in range(0, L-1, 2):
                Gamma, lambda_list, w = updateBond(Gamma, lambda_list, j, U_list, chi, w)
            # Half-step even bonds
            for j in range(1, L-1, 2):
                Gamma, lambda_list, w = updateBond(Gamma, lambda_list, j, U_half_list, chi, w)
    
            Sz_all[:, t_idx] = local_expectation_values(Gamma, lambda_list, Sz)
            S_all[:, t_idx] = entanglement_entropy(lambda_list)
            w_list[t_idx] = w
        
        """
        # Store results
        S_center_t_storage[case_label] = S_all[center_bond_idx, :]
        Sz_all_storage[case_label] = Sz_all.copy()
        S_all_storage[case_label] = S_all.copy()
        w_list_storage[case_label] = w_list.copy()
        """
        # 4. Store Results
        # Save center entropy for THIS chi (for the comparison plot)
        S_center_t_storage[case_label][chi] = S_all[center_bond_idx, :].copy()
        
        # Save the full lattice data (Overwrites previous chi, keeping the best/last one for heatmaps)
        Sz_all_storage[case_label] = Sz_all.copy()
        S_all_storage[case_label] = S_all.copy()
        w_list_storage[case_label] = w_list.copy()

    print(f"Simulation for {case_label} finished. Time elapsed: {time.time()-t_start_total:.2f} seconds.")

# --- PLOTTING -------------------------------------------------------------
# Colormap
cmap = LinearSegmentedColormap.from_list("blue_white_red", ["blue", "white", "red"])

# <Sz(x,t)> heatmaps
for case_label in cases.keys():
    Sz_all = Sz_all_storage[case_label]
    plt.figure(figsize=(8,6), dpi=500)
    plt.imshow(Sz_all.T, aspect='auto', origin='upper',
               extent=[1,L,time_steps[-1],time_steps[0]],
               cmap=cmap, vmin=-0.5, vmax=0.5)
    plt.xlabel('Position')
    plt.ylabel('Time')
    plt.title(fr'$\langle S_z(x,t) \rangle$ for {case_label}')
    plt.colorbar(label=r'$\langle S_z\rangle$')
    plt.grid(False)
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

    plt.show()

# 2D Entanglement entropy for all cases
for case_label in cases.keys():
    S_all = S_all_storage[case_label]
    plt.figure(figsize=(8,6), dpi=500)
    plt.imshow(S_all.T, aspect='auto', origin='upper', extent=[1,L,time_steps[-1],time_steps[0]],
               cmap='inferno')
    plt.xlabel('Position')
    plt.ylabel('Time')
    plt.title(f'Entanglement entropy S(x,t) for {case_label}')
    plt.colorbar(label=r'Entropy $S$')
    plt.grid(False)
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

    plt.show()

# Discarded weight ω(t)
for case_label in cases.keys():
    plt.figure(figsize=(8,6), dpi=500)
    plt.plot(time_steps, w_list_storage[case_label], label=case_label)
    plt.xlabel('Time t')
    plt.ylabel(r'Discarded weight $\omega$')
    plt.title('Accumulated discarded weight vs time')
    plt.grid(False)
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

    plt.show()

"""
# Entanglement entropy S(t) at center bond
for label, S_t in S_center_t_storage.items():
    plt.figure(figsize=(8,6), dpi=500)
    
    # Graficar los datos originales
    plt.plot(time_steps, S_t, label=label)
    
    # Ajuste de regresión lineal
    coef = np.polyfit(time_steps, S_t, 1)  # coef[0] = A, coef[1] = B
    linear_fit = np.polyval(coef, time_steps)
    
    # Crear label con la ecuación S(t) = A t + B
    A, B = coef
    eq_label = rf'Linear Fit: $S(t)={A:.3f} t + {B:.3f}$'
    
    # Graficar la línea de regresión
    plt.plot(time_steps, linear_fit, label=eq_label, linestyle=':')
    
    # Etiquetas y estilo
    plt.xlabel(r't')
    plt.ylabel(r'Entanglement Entropy $S(t)$')
    plt.title(r'Entanglement Growth over time at the Center Bond')
    plt.legend()
    plt.grid(False)
    plt.show()

# ---

# Entanglement entropy S(ln[t]) at center bond with Trend Lines
t_mask = time_steps > 0
log_t = np.log(time_steps[t_mask])


for label, S_t in S_center_t_storage.items():
    plt.figure(figsize=(8,6), dpi=500)
    
    # Filtrar S(t) para t > 0
    S_t_filtered = S_t[t_mask]
    
    # ----------------------------------------------------
    # 🌟 Linear Regression in log-time region
    # ----------------------------------------------------
    ln_t_min = 1.1 
    ln_t_max = 3.2 
    
    fit_mask = (log_t >= ln_t_min) & (log_t <= ln_t_max)
    
    if np.any(fit_mask):
        # Linear regression: fit_params = [slope, intercept]
        fit_params = np.polyfit(log_t[fit_mask], S_t_filtered[fit_mask], 1)
        slope, intercept = fit_params
        
        # Trend line for the full log_t range
        S_fit = slope * log_t + intercept
        
        # Plot fitted line with equation in legend
        eq_label = rf'Linear Fit: $S(t) = {slope:.3f} \ln t + {intercept:.3f}$'
        plt.plot(log_t, S_fit, label=eq_label, linestyle=':')
    
    # Plot actual S(t) data
    plt.plot(log_t, S_t_filtered, label=label)
    
    # Labels, title, legend
    plt.xlabel(r'$\ln t$')
    plt.ylabel(r'Entanglement Entropy $S(t)$')
    plt.title(r'Entanglement Growth over $\ln t$ at Center Bond')
    plt.legend()
    plt.grid(False)
    plt.show()
"""

# --- PLOTTING (S(t) Comparison) -------------------------------------------

# Entanglement entropy S(t) at center bond: One figure per Case
for case_label, chi_data_dict in S_center_t_storage.items():
    plt.figure(figsize=(8,6), dpi=500)
    
    # Loop over the chi values stored for this case
    # Sort chi keys to ensure legend order
    for chi in sorted(chi_data_dict.keys()):
        S_t = chi_data_dict[chi]
        plt.plot(time_steps, S_t, label=rf'$\chi={chi}$')

    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Entanglement Entropy $S(t)$')
    plt.title(f'Entanglement Growth: {case_label}')
    plt.legend(title=r'Bond Dim $\chi$')
    plt.grid(False)
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

    plt.show()

# Entanglement entropy S(ln t) with Trend Lines
t_mask = time_steps > 0
log_t = np.log(time_steps[t_mask])

for case_label, chi_data_dict in S_center_t_storage.items():
    plt.figure(figsize=(8,6), dpi=500)
    
    # We only fit the LARGEST chi to keep the plot clean
    max_chi = max(chi_data_dict.keys())
    
    for chi in sorted(chi_data_dict.keys()):
        S_t = chi_data_dict[chi]
        S_filtered = S_t[t_mask]
        
        # Plot data
        plt.plot(log_t, S_filtered, label=rf'$\chi={chi}$')
        """
        # Add trend line ONLY for the largest chi
        if chi == max_chi:
            ln_t_min, ln_t_max = 1.1, 3.2 
            fit_mask = (log_t >= ln_t_min) & (log_t <= ln_t_max)
            if np.any(fit_mask):
                fit_params = np.polyfit(log_t[fit_mask], S_filtered[fit_mask], 1)
                slope, intercept = fit_params
                S_fit = slope * log_t + intercept
                eq_label = rf'Fit (max $\chi$): ${slope:.2f} \ln t + {intercept:.2f}$'
                plt.plot(log_t, S_fit, 'k:', label=eq_label)
        """
    plt.xlabel(r'$\ln t$')
    plt.ylabel(r'Entanglement Entropy $S(t)$')
    plt.title(f'Entanglement vs $\ln t$: {case_label}')
    plt.legend(title=r'Bond Dimension $\chi$')
    plt.grid(False)
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

    plt.show()
    

# --- PLOT: Compare S(t) across ALL Cases (using max chi) ------------------
plt.figure(figsize=(8, 6), dpi=500)

# Iterate through all cases
for case_label, chi_data_dict in S_center_t_storage.items():
    # Find the largest chi available for this specific case
    max_chi = max(chi_data_dict.keys())
    
    # Retrieve the data
    S_t_best = chi_data_dict[max_chi]
    
    # Plot
    plt.plot(time_steps, S_t_best, label=f"{case_label} ($\chi={max_chi}$)")

plt.xlabel(r'Time $t$', fontsize=12)
plt.ylabel(r'Entanglement Entropy $S(t)$', fontsize=12)
plt.title(r'Entanglement Growth over time at the Center Bond across all Cases', fontsize=14)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

plt.show()