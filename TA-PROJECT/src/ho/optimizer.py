
import numpy as np
import math
import time

def levy_flight(n, m, beta=1.5):
    """
    Fungsi Levy Flight (Translasi dari levy.m)
    Digunakan pada Fase 2 (Predator) untuk lompatan eksplorasi ekstrem.
    """
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    
    z = u / (np.abs(v) ** (1 / beta))
    return z

def repair_solution(X, lowerbound, upperbound):
    """
    [INJEKSI KRIPTOGRAFI KHUSUS TUGAS AKHIR]
    Mencegah HO memasukkan parameter 'ilegal' ke dalam fungsi enkripsi.
    X = [r_min, eps, T0, Q]
    """
    # 1. Pastikan tidak melewati batas dasar
    X = np.clip(X, lowerbound, upperbound)
    
    # 2. Pembulatan (T0 dan Q WAJIB integer)
    X[2] = np.round(X[2])
    X[3] = np.round(X[3])
    
    # 3. Aturan Interval Chaos: r_min < 4.0 - eps
    r_min = X[0]
    eps = X[1]
    r_max = 4.0 - eps
    
    if r_min >= r_max:
        # Jika melanggar, paksa r_min mundur sedikit di bawah r_max
        X[0] = r_max - 1e-4 
        
    # Pastikan lagi tidak keluar batas setelah direparasi
    X = np.clip(X, lowerbound, upperbound)
    return X

def get_alfa(I1, I2, Ip1, dimension):
    """Fungsi pembantu untuk Fase 1 (Translasi dari variabel sel Alfa di MATLAB)"""
    choice = np.random.randint(1, 6)
    if choice == 1:
        return I2 * np.random.rand(dimension) + (1 - Ip1[0])
    elif choice == 2:
        return 2 * np.random.rand(dimension) - 1
    elif choice == 3:
        return np.random.rand(dimension)
    elif choice == 4:
        return I1 * np.random.rand(dimension) + (1 - Ip1[1])
    else:
        return np.random.rand() * np.ones(dimension)

def run_ho(SearchAgents, Max_iterations, lowerbound, upperbound, fitness_func):
    """
    Algoritma Hippopotamus Optimization (HO) Utama
    Translasi dari HO.m
    """
    dimension = len(lowerbound)
    lowerbound = np.array(lowerbound)
    upperbound = np.array(upperbound)
    
    # 1. Inisialisasi Populasi
    X = np.zeros((SearchAgents, dimension))
    fit = np.zeros(SearchAgents)
    
    for i in range(SearchAgents):
        X[i, :] = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)
        X[i, :] = repair_solution(X[i, :], lowerbound, upperbound)
        fit[i] = fitness_func(X[i, :])
        
    fbest = float('inf')
    Xbest = np.zeros(dimension)
    HO_curve = np.zeros(Max_iterations)
    
    start_time = time.time()
    
    # 2. Main Loop (t dimulai dari 1 agar tidak error saat dibagi t)
    for t in range(1, Max_iterations + 1):
        
        # Update Kandidat Terbaik
        best_idx = np.argmin(fit)
        best_score = fit[best_idx]
        
        if best_score < fbest:
            fbest = best_score
            Xbest = X[best_idx, :].copy()
            
        Dominant_hippopotamus = Xbest.copy()
        
        # --- PHASE 1: Posisi Sungai (Exploration) ---
        half_pop = int(SearchAgents / 2)
        for i in range(half_pop):
            I1 = np.random.randint(1, 3)
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, 2)
            
            RandGroupNumber = np.random.randint(1, SearchAgents + 1)
            RandGroup = np.random.permutation(SearchAgents)[:RandGroupNumber]
            
            if len(RandGroup) > 1:
                MeanGroup = np.mean(X[RandGroup, :], axis=0)
            else:
                MeanGroup = X[RandGroup[0], :]
                
            A = get_alfa(I1, I2, Ip1, dimension)
            B = get_alfa(I1, I2, Ip1, dimension)
            
            # Sub-Fase 1.1
            X_P1 = X[i, :] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i, :])
            X_P1 = repair_solution(X_P1, lowerbound, upperbound)
            F_P1 = fitness_func(X_P1)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1
                
            # Sub-Fase 1.2
            T = math.exp(-t / Max_iterations)
            X_P2 = np.zeros(dimension)
            if T > 0.6:
                X_P2 = X[i, :] + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i, :] + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = lowerbound + np.random.rand() * (upperbound - lowerbound)
                    
            X_P2 = repair_solution(X_P2, lowerbound, upperbound)
            F_P2 = fitness_func(X_P2)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        # --- PHASE 2: Pertahanan thd Predator (Exploration) ---
        for i in range(half_pop, SearchAgents):
            predator = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)
            F_HL = fitness_func(predator)
            
            distance2Leader = np.abs(predator - X[i, :])
            b = np.random.uniform(2, 4)
            c = np.random.uniform(1, 1.5)
            d = np.random.uniform(2, 3)
            l = np.random.uniform(-2 * math.pi, 2 * math.pi)
            
            RL = 0.05 * levy_flight(1, dimension, 1.5)[0]
            
            X_P3 = np.zeros(dimension)
            if fit[i] > F_HL:
                # Hindari pembagian dengan nol dengan menambahkan epsilon kecil
                X_P3 = RL * predator + (b / (c - d * math.cos(l))) * (1 / (distance2Leader + 1e-10))
            else:
                X_P3 = RL * predator + (b / (c - d * math.cos(l))) * (1 / (2 * distance2Leader + np.random.rand(dimension) + 1e-10))
                
            X_P3 = repair_solution(X_P3, lowerbound, upperbound)
            F_P3 = fitness_func(X_P3)
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3

        # --- PHASE 3: Menghindar (Exploitation) ---
        for i in range(SearchAgents):
            LO_LOCAL = lowerbound / t
            HI_LOCAL = upperbound / t
            
            choice = np.random.randint(1, 4)
            if choice == 1: D = 2 * np.random.rand(dimension) - 1
            elif choice == 2: D = np.random.rand() * np.ones(dimension)
            else: D = np.random.randn() * np.ones(dimension)
            
            X_P4 = X[i, :] + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))
            X_P4 = repair_solution(X_P4, lowerbound, upperbound)
            
            F_P4 = fitness_func(X_P4)
            if F_P4 < fit[i]:
                X[i, :] = X_P4
                fit[i] = F_P4
                
        # Simpan Best So Far
        HO_curve[t-1] = fbest
        print(f"[HO] Iterasi {t}/{Max_iterations} | Best Fitness: {fbest:.6f}")

    eval_time = time.time() - start_time
    print(f"\n[SELESAI] Optimasi selesai dalam {eval_time:.2f} detik.")
    return fbest, Xbest, HO_curve