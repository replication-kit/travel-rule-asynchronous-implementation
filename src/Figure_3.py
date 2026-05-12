# %%
# ============================================
# Figure 3: Robustness of transitional failure
# Full replacement version (aligned with Figure_2.py)
# ============================================

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Scenario parameters (same as Figure_2.py)
# ----------------------------
def scenario_params(scenario: str):
    if scenario == "S":  # synchronized
        Tj = np.array([6, 6, 6, 6, 6, 6])
        rho = np.array([0.06, 0.06, 0.05, 0.05, 0.04, 0.04])
    elif scenario == "A":  # asynchronous
        Tj = np.array([6, 6, 18, 18, 36, 36])
        rho = np.array([0.06, 0.06, 0.05, 0.05, 0.04, 0.04])
    elif scenario == "Aprime":  # asynchronous + paper compliance
        Tj = np.array([6, 6, 18, 18, 36, 36])
        rho = np.array([0.06, 0.06, 0.05, 0.05, 0.015, 0.015])
    else:
        raise ValueError("Unknown scenario")
    return Tj, rho


# ----------------------------
# Cross-border weights and sunrise gap (same as Figure_2.py)
# ----------------------------
def compute_pair_weights(J: int = 6) -> np.ndarray:
    w = np.zeros((J, J))
    weights = {}

    def add(j, k, val):
        weights[(j, k)] = val

    core = [0, 1]
    mid = [2, 3]
    lag = [4, 5]

    add(0, 1, 3.0)
    for j in core:
        for k in mid:
            add(min(j, k), max(j, k), 2.0)
        for k in lag:
            add(min(j, k), max(j, k), 2.0)
    add(2, 3, 1.0)
    for j in mid:
        for k in lag:
            add(min(j, k), max(j, k), 1.0)
    add(4, 5, 0.5)

    total = sum(weights.values())
    for (j, k), val in weights.items():
        w[j, k] = val / total
        w[k, j] = w[j, k]
    return w


def compute_X(L: np.ndarray, R: np.ndarray, w: np.ndarray) -> float:
    """Aggregate international interoperability X_t."""
    X = 0.0
    J = len(L)
    for j in range(J):
        for k in range(j + 1, J):
            if L[j] == 1 and L[k] == 1:
                X += w[j, k] * min(R[j], R[k])
    return float(np.clip(X, 0.0, 1.0))


# ----------------------------
# Vectorized logit choice (exact probabilities)
# ----------------------------
def sample_choices(costA: np.ndarray, costB: np.ndarray, costC: float, lam: float, rng) -> np.ndarray:
    """
    Sample choices in {0,1,2} for each agent with logit probabilities:
      P(c) ∝ exp(-lam * cost_c)
    costA, costB are arrays of length N; costC is scalar.
    """
    valsA = -lam * costA
    valsB = -lam * costB
    valsC = -lam * costC  # scalar

    # Stabilize exponentials using per-agent max (broadcasting handles scalar valsC)
    m = np.maximum(np.maximum(valsA, valsB), valsC)
    eA = np.exp(valsA - m)
    eB = np.exp(valsB - m)
    eC = np.exp(valsC - m)
    s = eA + eB + eC

    pA = eA / s
    pB = eB / s
    # pC = eC / s

    u = rng.random(costA.shape[0])
    choices = np.empty_like(costA, dtype=int)
    choices[u < pA] = 0
    choices[(u >= pA) & (u < pA + pB)] = 1
    choices[u >= pA + pB] = 2
    return choices


# ----------------------------
# Core ABM run (aligned with Figure_2.py; returns Y only)
# ----------------------------
def run_one(
    scenario: str,
    seed: int,
    *,
    T: int = 60,
    N: int = 100,
    J: int = 6,
    pi_cross: float = 0.35,
    lam: float = 5.0,
    pA0: float = 0.18,
    pB0: float = 0.08,
    pC0: float = 0.02,
    a_s: float = 0.40,
    b_s: float = 0.50,
    F: float = 6.0,
    kA: float = 0.20,
    kB: float = 0.05,
    kC_base: float = 0.80,
    kC_min: float = 0.02,
    kappa: float = 0.10,
    Tj_override=None,
    rho_override=None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    Tj, rho = scenario_params(scenario)
    if Tj_override is not None:
        Tj = np.array(Tj_override, dtype=int)
    if rho_override is not None:
        rho = np.array(rho_override, dtype=float)

    L = np.zeros(J)
    R = np.zeros(J)
    w = compute_pair_weights(J)

    Y = np.zeros(T)
    N_C_prev = 0

    for t in range(T):
        # Update adoption status and interoperability
        for j in range(J):
            if t >= Tj[j]:
                L[j] = 1.0
                R[j] = min(1.0, R[j] + rho[j])

        # Sunrise gap (same definition as Figure_2.py)
        s_gap = 1.0 - compute_X(L, R, w)

        # Endogenous P2P friction (same functional form as Figure_2.py)
        kC = max(kC_min, kC_base - kappa * np.log(1.0 + N_C_prev))

        # Cross-border indicator (agent-specific)
        cross = rng.random(N) < pi_cross

        # Detection probabilities
        # (For A/B: cross-border only; domestic stays at baseline pA0/pB0)
        pA = np.clip(pA0 - a_s * s_gap * cross, 0.0, 1.0)
        pB = np.clip(pB0 - b_s * s_gap * cross, 0.0, 1.0)
        pC = np.full(N, pC0)

        # Expected costs
        costA = pA * F + kA
        costB = pB * F + kB
        costC = pC0 * F + kC  # scalar

        # Channel choice
        choices = sample_choices(costA, costB, costC, lam, rng)

        # Detection realization
        p_true = np.where(choices == 0, pA, np.where(choices == 1, pB, pC))
        caught = rng.random(N) < p_true

        # Successful illicit transactions
        Y[t] = N - caught.sum()

        # Update lagged P2P usage
        N_C_prev = int(np.sum(choices == 2))

    return Y


# ----------------------------
# Transitional mean Y over a window
# ----------------------------
def transitional_mean_Y(
    scenario: str,
    *,
    Rruns: int,
    t1: int = 6,
    t2: int = 35,
    Tj_override=None,
    rho_override=None,
    seed0: int = 100000,
) -> float:
    vals = []
    for r in range(Rruns):
        Y = run_one(
            scenario,
            seed=seed0 + r,
            Tj_override=Tj_override,
            rho_override=rho_override,
        )
        vals.append(Y[t1 : t2 + 1].mean())
    return float(np.mean(vals))


# ----------------------------
# Figure 3
# ----------------------------
def make_figure3(
    *,
    Rruns: int = 200,
    # Panel-specific averaging windows
    # Panel (a): pre-lag window (baseline lag adoption is at t=36)
    t1_a: int = 6,
    t2_a: int = 35,
    # Panel (b): post-lag window to capture implementation-speed effects
    t1_b: int = 36,
    t2_b: int = 59,
    deltaT_grid=(0, 6, 12, 18, 24, 30),
    rho_lag_grid=(0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    # Common-random-number seeds (panel-specific)
    seed_panel_a: int = 200000,
    seed_panel_b: int = 300000,
    savepath: str = "Figure3.png",
):
    # Baseline (synchronized) computed using the same random streams as each panel
    YS_a = transitional_mean_Y("S", Rruns=Rruns, t1=t1_a, t2=t2_a, seed0=seed_panel_a)
    YS_b = transitional_mean_Y("S", Rruns=Rruns, t1=t1_b, t2=t2_b, seed0=seed_panel_b)

    base_Tj_A, base_rho_A = scenario_params("A")
    base_Tj_Ap, base_rho_Ap = scenario_params("Aprime")

    # ---- Panel (a): Adoption delay (vary T_lag, hold T_mid fixed) ----
    YA, YAp = [], []
    for dT in deltaT_grid:
        TjA = base_Tj_A.copy()
        TjA[4:] = TjA[0] + dT  # lagging adopt at core time + dT

        TjAp = base_Tj_Ap.copy()
        TjAp[4:] = TjAp[0] + dT

        YA.append(
            transitional_mean_Y(
                "A",
                Rruns=Rruns,
                t1=t1_a,
                t2=t2_a,
                Tj_override=TjA,
                rho_override=base_rho_A,
                seed0=seed_panel_a,
            )
            - YS_a
        )
        YAp.append(
            transitional_mean_Y(
                "Aprime",
                Rruns=Rruns,
                t1=t1_a,
                t2=t2_a,
                Tj_override=TjAp,
                rho_override=base_rho_Ap,
                seed0=seed_panel_a,
            )
            - YS_a
        )

    # ---- Panel (b): Implementation speed (vary rho_lag only for A′; keep A at baseline) ----
    YA_const = (
        transitional_mean_Y(
            "A",
            Rruns=Rruns,
            t1=t1_b,
            t2=t2_b,
            Tj_override=base_Tj_A,
            rho_override=base_rho_A,
            seed0=seed_panel_b,
        )
        - YS_b
    )
    YA_rho = [YA_const] * len(rho_lag_grid)

    YAp_rho = []
    for rl in rho_lag_grid:
        rhoAp = base_rho_Ap.copy()
        rhoAp[4:] = rl  # vary only lagging jurisdictions
        YAp_rho.append(
            transitional_mean_Y(
                "Aprime",
                Rruns=Rruns,
                t1=t1_b,
                t2=t2_b,
                Tj_override=base_Tj_Ap,
                rho_override=rhoAp,
                seed0=seed_panel_b,
            )
            - YS_b
        )

    # ---- Plot ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Panel (a): ○ and △
    axs[0].scatter(
        deltaT_grid,
        YA,
        label="A",
        marker="o",
        facecolors="none",
        edgecolors="black",
        s=70,
    )
    axs[0].scatter(
        deltaT_grid,
        YAp,
        label="A′",
        marker="^",
        facecolors="none",
        edgecolors="black",
        s=70,
    )
    axs[0].set_xlabel("Adoption delay (ΔT = T_lag − T_core)")
    axs[0].set_ylabel("Δ mean illicit activity (vs synchronized)")
    axs[0].set_title("(a): Adoption delay")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Panel (b): ○ and △
    axs[1].scatter(
        rho_lag_grid,
        YA_rho,
        label="A",
        marker="o",
        facecolors="none",
        edgecolors="black",
        s=70,
    )
    axs[1].scatter(
        rho_lag_grid,
        YAp_rho,
        label="A′",
        marker="^",
        facecolors="none",
        edgecolors="black",
        s=70,
    )
    axs[1].set_xlabel("Implementation speed in lagging jurisdictions (ρ_lag)")
    axs[1].set_title("(b): Implementation speed")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    make_figure3()