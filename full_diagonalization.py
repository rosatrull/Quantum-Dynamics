#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 13:25:39 2025

@author: rosatrullcastanyer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Configuración para usar LaTeX
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# --- CONSTRUCT THE COMPLET HAMILTONIAN MATRIX IN FOCK SPACE --------------------------------------------------
# H = J ∑ S_i S_{i+1} - h ∑ S_i^z
# H = J ∑ 1/2 (S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + S_i^zS_{i+1}^z - h ∑ S_i^z

# The basis is implicitly defined here. 
# range(n_states) generates the sequence 0, 1, ..., 2^N - 1

def get_hamiltonian(N, J, h, periodic):
    """
    Constructs the Heisenberg Hamiltonian Matrix element by element.
    """
    dim = 2**N
    # Initialize the whole matrix with 0.0
    H = np.zeros((dim, dim))
    
    print(f"Building Matrix for N={N}...")
    
    # Loop over every possible spin configuration (basis state)
    for state in range(dim):
        
        # --- Part A: Diagonal Elements ---------------------------------------
        # H_aa = ∑_i  J S_i^z * S_{i+1}^z − h ∑_i  S_i^z
        
        H_aa = 0 # We initialize diagonal energies to zero
        
        # Loop over sites i to calculate sum(Si^z * Si+1^z)
        for i in range(N):
            # Find the neighbor 'j'
            if periodic: 
                j = (i + 1) % N # periodic boundary conditions. The neighbor of site N is intial site 0
            else:
                j = i + 1
                if j >= N: continue # Stop at the edge for Open Chain
            
            s_i = (state >> i) & 1  
            s_j = (state >> j) & 1
            
            # >> is the right shift operator, takes the binary number and shifts all bits to the right by i positions
            # & is the Bitwise and operator compares the last bit with 1. If the last bit is 1, the result is 1. If the last bit is 0, the result is 0
            
            
            # Convert 0/1 to -0.5/+0.5
            # 0 represents spin down ↓
            # 1 represents spin up ↑
            Sz_i = 0.5 if s_i == 1 else -0.5
            Sz_j = 0.5 if s_j == 1 else -0.5
            
            # Add interaction term J * Sz * Sz
            H_aa += J * Sz_i * Sz_j
            
            #Since we are already visiting site 'i', let's just add its magnetic energy now.
            # We subtract h * Sz_i from the diagonal energy immediately.
            H_aa -= h * Sz_i
        
        # Store on diagonal
        H[state, state] = H_aa

        # --- Part B: Off-Diagonal Elements -----------------------------------
        # The term J/2 * (S+S- + S-S+) exchanges spins: Up-Down becomes Down-Up
        
        for i in range(N): 
            if periodic: 
                j = (i + 1) % N
            else:
                j = i + 1
                if j >= N: continue            
            
            # If site i is Down (↓) and site i+1 is Up (↑), it flips them: ↓↑ → ↑↓.
            # If they are already parallel (↑↑ or ↓↓), this term gives zero
            # finds every pair where a flip is possible
            
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            
            # Check if spins are different (only then can they flip)
            if bit_i != bit_j:
                # We found an Up-Down or Down-Up pair!
                # Create the NEW state index by flipping bits i and j
                # The XOR operator (^) with a mask does this perfectly
                flip_mask = (1 << i) | (1 << j)
                new_state = state ^ flip_mask    # If the mask is 1, flip the bit. If the mask is 0, leave it alone.
                
                # Update the matrix
                H[state, new_state] += J / 2.0
                
    return H

# --- DIAGONALIZE H --------------------------------------------------------

def diagonalization(H):
    """Diagonalizes H to find Eigenvalues (Energies) and Eigenvectors"""
    evals, evecs = np.linalg.eigh(H)
    # evecs is the matrix woth eigenvectors as columns evecs[:, i] is the eigenvector of the eigenvalue i
    # the diagonal matrix would be D = np.diag(evals)
    return evals, evecs

# --- CALCULATE OBSERVABLES -----------------------------------------------

# THERMODYNAMICS
def calculate_observables(evals, evecs, beta, N):
    """Calculates Energy and Magnetization using Boltzmann statistics"""
    # Boltzmann Factors (Probabilities)
    """
    # We shift energies by min(E) to avoid number overflow
    E_min = evals[0]
    weights = np.exp(-beta * (evals - E_min))
    """
    # Z = tr exp(-βH)
    # In linear algebra, the trace of any function of a matrix is equal to the sum of that function applied to its eigenvalues.
    # Z = tr exp(−βH)= ∑_n exp(−βE_n)
    weights = np.exp(-beta * evals)
    Z = np.sum(weights) # Partition function
    
    # --- Average Energy ------------------------------------------------------
    # To find the Average Energy ⟨E⟩, you sum up every possible energy multiplied by the probability of being in that state
    # p = exp(−βE_i) / ∑_n exp(−βE_n)
    avg_E = np.sum(evals * weights) / Z
    
    # --- Magnetization per site < M / N > -------------------------------------------------
    dim = 2**N
  
    # We calculate M for every basis state first
    # M = ∑_i S_i^z
    M_diag = np.zeros(dim)
    for s in range(dim):
        m_val = 0    # reset the magnetization counter for the current state ∣s⟩
        
        # Check the spin at site k
        for k in range(N):
            m_val += 0.5 if ((s >> k) & 1) else -0.5
        M_diag[s] = m_val  # Store the total magnetization for state ∣s⟩
        
    # Expectation value in Energy basis
    # <n|M|n> = sum of |coefficient|^2 * M_state (this comes from | n > = ∑_s c_s | s > where c_s is the coefficient (amplitude) < s | n >
    # evecs[s, n] = < s | n >
    M_energy_basis = np.sum((evecs**2).T * M_diag, axis=1)  # axis=1 is crucial because it tells the computer: Sum up all the basis states to finish the recipe for ONE energy state
    
    avg_M = np.sum(M_energy_basis * weights) / Z
    avg_M_per_site = avg_M / N
    
    # --- Susceptibility per site 𝜒/N = β (< M^2 > - < M >^2)/N ----------------------------------
    # Remember M is diagonal --> M^2 is also diagonal
    M2_energy_basis = np.sum((evecs**2).T * (M_diag**2), axis=1)
    avg_M2 = np.sum(M2_energy_basis * weights) / Z
    
    chi = beta * (avg_M2 - avg_M**2)
    chi_per_site = chi / N
    
    # --- Ground state energy per site ----------------------------------------
    # Since numpy.linalg.eigh returns eigenvalues sorted from smallest to largest,
    # evals[0] is guaranteed to be the Ground State Energy (E_gs).
    E_gs = evals[0]
    E_gs_per_site = E_gs / N
    
    return avg_E, avg_M_per_site, chi_per_site, E_gs_per_site

# --- RUN THE EXERCICE: PLOTING ----------------------------------------

def run_thermodynamics():
    print("--- Starting Calculations ---")

    # --- TOTAL ENERGY --- 
    N = 8
    J = 1.0
    beta = 2.0
    h = 0.5
    print(f"\n1. Calculating Total Energy for N={N}, J={J}, h = {h}, β = {beta} ...")
    
    H = get_hamiltonian(N, J, h, periodic=True)
    evals, evecs = diagonalization(H)
    E_val, _, _, _ = calculate_observables(evals, evecs, beta, N)
    print(f"Your Result:   {E_val:.4f} (Target Energy: -2.867)")

    # --- PLOT A: Magnetization vs Field (h) ---
    print("\n2. Generating Plot A (Magnetization per site vs Magentic Field)...")
    
    h_list = np.linspace(0, 2.0, 20)
    # We want beta*J = 40. Let's fix J=1, so beta=40
    beta = 40  # try 5 o see it smoth line not steps
    J = 1
    
    plt.figure(figsize=(8,6), dpi=500)
    for N_val in [7, 8]: # Vary N
        m_list = []
        for h in h_list:
            # Re-build matrix for new h
            H = get_hamiltonian(N_val, J, h, periodic=True)
            evals, evecs = diagonalization(H)
            _, m_site, _, _ = calculate_observables(evals, evecs, beta, N_val)
            m_list.append(m_site)
        plt.plot(h_list, m_list, label=f'N={N_val}')

    plt.title(rf'Magnetization vs Field ($\beta J={beta*J}$)')
    plt.xlabel('Magnetic Field h')
    plt.ylabel(r'Magnetization $\langle M \rangle / N$')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)
    
    print("\n2. Generating Plot A.2 (Magnetization per site vs Magentic Field for different betas)...")
    
    h_list = np.linspace(0, 2.0, 20)
    # We want beta*J = 40. Let's fix J=1, so beta=40
    beta = [40, 5]  # try 5 o see it smoth line not steps
    
    plt.figure(figsize=(8,6), dpi=500)
    for b in beta: # Vary beta
        N = 8
        J = 40/b
        m_list = []
        for h in h_list:
            # Re-build matrix for new h
            H = get_hamiltonian(N, J, h, periodic=True)
            evals, evecs = diagonalization(H)
            _, m_site, _, _ = calculate_observables(evals, evecs, b, N)
            m_list.append(m_site)
        plt.plot(h_list, m_list, label=fr'$\beta$={b}')

    plt.title('Magnetization vs Field')
    plt.xlabel('Magnetic Field h')
    plt.ylabel(r'Magnetization $\langle M \rangle / N$')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

    # --- PLOT B: Susceptibility vs Temperature (beta) ---
    print("3. Generating Plot B (Susceptibility per site vs Beta)...")
    
    beta_values = np.linspace(0.1, 5.0, 50)
    J = 1.0
    h = 0 # Zero field
    
    plt.figure(figsize=(8,6), dpi=500)
    for N_val in [7, 8]:
        # H is independent of beta, so calculate it ONCE per N
        H = get_hamiltonian(N_val, J, h, periodic=True)
        evals, evecs = diagonalization(H)
        
        chi_list = []
        for b_val in beta_values:
            _, _, chi_site, _ = calculate_observables(evals, evecs, b_val, N_val)
            chi_list.append(chi_site)
        plt.plot(beta_values, chi_list, label=f'N={N_val}')

    plt.title(f'Susceptibility vs Beta (h={h})')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'Susceptibility per site $\chi/N$')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)


    # --- PLOT C: Ground State Energy vs System Size ---
    print("4. Generating Plot C (GS Energy vs N)...")
    
    N_sizes = range(2, 13) # 2 to 12
    beta = 30
    J = 1.0
    h = 0
    
    gs_energies = []
    for N_val in N_sizes:
        H = get_hamiltonian(N_val, J, h, periodic=True)
        evals, evecs = diagonalization(H)
        
        # We don't need beta for E_gs, but the function asks for it. 
        _, _, _, E_gs_site = calculate_observables(evals, evecs, beta, N=N_val)
        gs_energies.append(E_gs_site)
    
    plt.figure(figsize=(8,6), dpi=500)
    plt.plot(N_sizes, gs_energies)
    plt.title('Ground State Energy per site as a function of sistem size')
    plt.xlabel('System Size N')
    plt.ylabel(r'$E_{gs} / N$')
    plt.grid(False)
    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)
    plt.show()
    print("Done!")

# run thermodynamics
run_thermodynamics()

def simulate_time_evolution():
    """
    Simulates a single particle propagating through an Open Chain.
    """
    N = 8
    J = 1
    h = 1
    t_max = 12
    dt = 0.1
    # Define which sites should be Spin Up
    spin_up_1 = [1]
    spin_up_sites = [s - 1 for s in spin_up_1] # remember python idexs starts with 0
    
    print(f"\n--- Starting Time Evolution (N={N}, Open Boundaries) ---")
    
    # 1. Hamiltonian with OPEN Boundary Conditions (periodic=False)
    # The particle needs to bounce off walls, not loop around.
    H = get_hamiltonian(N, J, h, periodic=False) 
    evals, evecs = diagonalization(H)
    
    # 2. Define Initial State |psi(0)>
    # Convert this list into the single integer index for the Fock state
    # We start with 0 (all down) and flip the bits for the sites in our list
    psi_0_index = 0
    for site in spin_up_sites:
        psi_0_index += (1 << site) # Sets the bit at position 'site' to 1
        
    # C. Create the state vector |psi(0)>
    psi_0 = np.zeros(2**N, dtype=complex)
    psi_0[psi_0_index] = 1.0
    
    # D. Visual Check (The arrow printing loop you wanted)
    # We reconstruct the arrows by reading the bits of psi_0_index
    spins_visual = []
    for n in range(N):
        # Check if bit 'n' is set in our index
        is_up = (psi_0_index >> n) & 1
        
        if is_up:
            spins_visual.append("↑") # Spin Up
        else:
            spins_visual.append("↓") # Spin Down

    # Print nicely formatted
    print("Initial state: " + "".join(spins_visual))

    # 3. Project onto Eigenbasis: c_n = <n | psi_0>
    coeffs_0 = evecs.T.dot(psi_0)
    
    # 4. Time Evolution Loop
    times = np.arange(0, t_max, dt)
    magnetization_map = np.zeros((len(times), N))
    
    for t_idx, t in enumerate(times):
        # Evolve coefficients: c_n(t) = c_n(0) * exp(-i * E_n * t)
        coeffs_t = coeffs_0 * np.exp(-1j * evals * t)
        
        # Reconstruct wavefunction: psi(t) = sum( c_n(t) * |n> )
        psi_t = evecs.dot(coeffs_t)
        
        # Calculate Local Density (Probability of Up spin at site k)
        prob_density = np.abs(psi_t)**2
        
        for site in range(N):
            # Sum prob of all states where 'site' is 1
            site_mag = 0.0
            for state_idx, prob in enumerate(prob_density):
                if (state_idx >> site) & 1:
                    site_mag += prob
            magnetization_map[t_idx, site] = site_mag

    # 5. Plotting
    plt.figure(figsize=(8,6), dpi=500)
    plt.imshow(magnetization_map, aspect='auto', origin='lower', extent=[0.5, N+0.5, 0, t_max], cmap='inferno')
    
    plt.colorbar(label=r'Spin Probability density of finding a particle $\langle n_i \rangle=\langle S_i^+ S_i^- \rangle$')
    plt.xlabel('Lattice Site')
    plt.ylabel(r'Time $\left(\hbar / J\right)$')
    plt.title(f'Time Evolution of Single Spin Flip (N={N})')
    # TICKS: Standard 1 to N
    plt.xticks(range(1, N + 1))
    
    # LIMITS: Force the plot to crop exactly at the data edges
    plt.xlim(0.5, N + 0.5)
    plt.ylim(0, t_max)
    
    #  Flips time so it runs top → bottom
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)
    plt.show()
  
# run time evolution
simulate_time_evolution()

