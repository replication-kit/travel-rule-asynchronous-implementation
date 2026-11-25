#!/usr/bin/env python3
"""
verify_model.py

Numerical verification for the revised model in:
"Implementation Gaps, Channel Substitution, and Hysteresis in Crypto-Asset Regulation"

The script verifies:
1. Unique-equilibrium and multiple-equilibrium regions.
2. The reduced steady-state equation.
3. Stability of low / middle / high equilibria.
4. The basin-boundary property of the unstable equilibrium.
5. Lemma 2: conditional maturity investment under fixed-average asymmetry.
6. Leader-versus-laggard comparative statics and Proposition 3 derivatives.
7. Fixed-weakest-link transition histories (fast leader vs synchronization).
8. The critical leader exponent and the two equilibrium-selection regions.
9. The no-hysteresis benchmark using the same updated transition paths.

This is a verification script, not a calibration exercise.
"""

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    # Generalized-cost parameters
    cC: float = 0.0
    cW_tilde: float = 1.0
    cP: float = 3.0
    phi: float = 1.0
    tau: float = 0.25
    lam: float = 5.3
    sigma: float = 1.0

    # P2P maturity dynamics
    delta: float = 0.10
    thetaW: float = 0.10
    thetaP: float = 0.13


def implementation_stats(e1: float, e2: float) -> Tuple[float, float, float]:
    """Return average implementation, asymmetry, and weakest-link implementation."""
    ebar = 0.5 * (e1 + e2)
    d = abs(e1 - e2)
    emin = min(e1, e2)
    return ebar, d, emin


def costs(e1: float, e2: float, p: float, par: Params) -> Tuple[float, float, float]:
    """Generalized costs (C, W, P)."""
    ebar, _, emin = implementation_stats(e1, e2)
    CC = par.cC
    CW = par.cW_tilde + par.phi * emin
    CP = par.cP + par.tau * ebar - par.lam * p
    return CC, CW, CP


def shares(e1: float, e2: float, p: float, par: Params) -> Tuple[float, float, float]:
    """Multinomial-logit transaction shares."""
    CC, CW, CP = costs(e1, e2, p, par)
    z = np.array([-CC / par.sigma, -CW / par.sigma, -CP / par.sigma], dtype=float)
    z -= np.max(z)
    w = np.exp(z)
    probs = w / w.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


def maturity_investment(e1: float, e2: float, p: float, par: Params) -> float:
    """Gross investment I_t = thetaW*xW + thetaP*xP."""
    _, xW, xP = shares(e1, e2, p, par)
    return par.thetaW * xW + par.thetaP * xP


def maturity_next(e1: float, e2: float, p: float, par: Params) -> float:
    """Law of motion for P2P maturity."""
    return (1.0 - par.delta) * p + maturity_investment(e1, e2, p, par)


def reduced_objects(par: Params) -> Dict[str, float]:
    """Compute r, psi0, psi1, A, and B after implementation is complete."""
    aC = exp(-par.cC / par.sigma)
    aW = exp(-(par.cW_tilde + par.phi) / par.sigma)
    r = aW / (aC + aW)
    psi0 = par.thetaW * r / par.delta
    psi1 = (par.thetaP - par.thetaW * r) / par.delta
    A = (
        -(par.cP + par.tau) / par.sigma
        - log(aC + aW)
        + par.lam * psi0 / par.sigma
    )
    B = par.lam * psi1 / par.sigma
    return {"aC": aC, "aW": aW, "r": r, "psi0": psi0, "psi1": psi1, "A": A, "B": B}


def H(x: float, B: float) -> float:
    return log(x / (1.0 - x)) - B * x


def equilibrium_residual(x: float, A: float, B: float) -> float:
    return H(x, B) - A


def bisection_root(
    f: Callable[[float], float], lo: float, hi: float,
    tol: float = 1e-12, max_iter: int = 500,
) -> float:
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError(f"Root not bracketed on [{lo}, {hi}].")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def find_equilibria(A: float, B: float) -> List[float]:
    """Find all interior equilibria using Proposition 1 to bracket roots."""
    eps = 1e-10
    f = lambda x: equilibrium_residual(x, A, B)

    if B <= 4.0:
        return [bisection_root(f, eps, 1.0 - eps)]

    disc = sqrt(1.0 - 4.0 / B)
    x_minus = 0.5 * (1.0 - disc)
    x_plus = 0.5 * (1.0 + disc)
    Hmax = H(x_minus, B)
    Hmin = H(x_plus, B)

    if A > Hmax:
        return [bisection_root(f, x_plus, 1.0 - eps)]
    if A < Hmin:
        return [bisection_root(f, eps, x_minus)]

    if abs(A - Hmax) < 1e-10:
        left = bisection_root(f, eps, x_minus)
        return sorted([left, x_minus])
    if abs(A - Hmin) < 1e-10:
        right = bisection_root(f, x_plus, 1.0 - eps)
        return sorted([x_plus, right])

    return [
        bisection_root(f, eps, x_minus),
        bisection_root(f, x_minus, x_plus),
        bisection_root(f, x_plus, 1.0 - eps),
    ]


def maturity_from_x(x: float, par: Params) -> float:
    obj = reduced_objects(par)
    return obj["psi0"] + obj["psi1"] * x


def stability_derivative(x: float, par: Params) -> float:
    B = reduced_objects(par)["B"]
    return 1.0 - par.delta + par.delta * B * x * (1.0 - x)


def verify_proposition_1(par: Params) -> Dict[str, object]:
    obj = reduced_objects(par)
    A = obj["A"]
    B = obj["B"]
    roots = find_equilibria(A, B)
    out: Dict[str, object] = {
        **obj,
        "roots": roots,
        "num_roots": len(roots),
        "residuals": [equilibrium_residual(x, A, B) for x in roots],
    }
    if B > 4.0:
        disc = sqrt(1.0 - 4.0 / B)
        x_minus = 0.5 * (1.0 - disc)
        x_plus = 0.5 * (1.0 + disc)
        out.update({
            "x_minus": x_minus,
            "x_plus": x_plus,
            "H_x_minus": H(x_minus, B),
            "H_x_plus": H(x_plus, B),
            "three_eq_condition": H(x_plus, B) < A < H(x_minus, B),
        })
    return out


def verify_proposition_2(par: Params) -> Dict[str, object]:
    p1 = verify_proposition_1(par)
    roots = p1["roots"]
    maturities = [maturity_from_x(x, par) for x in roots]
    derivatives = [stability_derivative(x, par) for x in roots]
    stable = [abs(gp) < 1.0 for gp in derivatives]
    return {"x": roots, "p": maturities, "Gprime": derivatives, "stable": stable}


def analytical_dI_dd(e1: float, e2: float, p: float, par: Params) -> float:
    xC, xW, xP = shares(e1, e2, p, par)
    return (
        par.phi / (2.0 * par.sigma) * xW
        * (par.thetaW * xC + (par.thetaW - par.thetaP) * xP)
    )


def numerical_dI_dd(ebar: float, d: float, p: float, par: Params, h: float = 1e-6) -> float:
    def I_of_d(dd: float) -> float:
        e1 = ebar + 0.5 * dd
        e2 = ebar - 0.5 * dd
        return maturity_investment(e1, e2, p, par)
    return (I_of_d(d + h) - I_of_d(d - h)) / (2.0 * h)


def p_hat(ebar: float, par: Params) -> float:
    if par.thetaP <= par.thetaW:
        return float("inf")
    ratio = par.thetaW / (par.thetaP - par.thetaW)
    return (
        par.sigma * log(ratio)
        - par.cC + par.cP + par.tau * ebar
    ) / par.lam


def analytical_dxW_de1(e1: float, e2: float, p: float, par: Params) -> float:
    _, xW, xP = shares(e1, e2, p, par)
    return par.tau / (2.0 * par.sigma) * xW * xP


def analytical_dxW_de2(e1: float, e2: float, p: float, par: Params) -> float:
    _, xW, xP = shares(e1, e2, p, par)
    return xW / par.sigma * (par.tau / 2.0 * xP - par.phi * (1.0 - xW))


def numerical_dxW_de1(e1: float, e2: float, p: float, par: Params, h: float = 1e-6) -> float:
    plus = shares(e1 + h, e2, p, par)[1]
    minus = shares(e1 - h, e2, p, par)[1]
    return (plus - minus) / (2.0 * h)


def numerical_dxW_de2(e1: float, e2: float, p: float, par: Params, h: float = 1e-6) -> float:
    plus = shares(e1, e2 + h, p, par)[1]
    minus = shares(e1, e2 - h, p, par)[1]
    return (plus - minus) / (2.0 * h)


def simulate(e1_path: np.ndarray, e2_path: np.ndarray, p0: float, par: Params) -> Dict[str, np.ndarray]:
    if len(e1_path) != len(e2_path):
        raise ValueError("e1_path and e2_path must have equal length.")
    T = len(e1_path)
    p = np.empty(T + 1)
    xC = np.empty(T)
    xW = np.empty(T)
    xP = np.empty(T)
    ebar = np.empty(T)
    d = np.empty(T)
    p[0] = p0
    for t in range(T):
        ebar[t], d[t], _ = implementation_stats(e1_path[t], e2_path[t])
        xC[t], xW[t], xP[t] = shares(e1_path[t], e2_path[t], p[t], par)
        p[t + 1] = maturity_next(e1_path[t], e2_path[t], p[t], par)
    return {
        "e1": np.asarray(e1_path), "e2": np.asarray(e2_path),
        "ebar": ebar, "d": d, "xC": xC, "xW": xW, "xP": xP, "p": p,
    }


def post_implementation_tail(p0: float, par: Params, periods: int = 1000) -> Dict[str, np.ndarray]:
    return simulate(np.ones(periods), np.ones(periods), p0, par)


def classify_A_B(A: float, B: float) -> str:
    if B <= 4.0:
        return "unique"
    disc = sqrt(1.0 - 4.0 / B)
    x_minus = 0.5 * (1.0 - disc)
    x_plus = 0.5 * (1.0 + disc)
    Hmax = H(x_minus, B)
    Hmin = H(x_plus, B)
    if Hmin < A < Hmax:
        return "three"
    if abs(A - Hmin) < 1e-10 or abs(A - Hmax) < 1e-10:
        return "saddle-node"
    return "unique"


def scan_reduced_parameter_region(A_values: np.ndarray, B_values: np.ndarray) -> Dict[str, int]:
    counts = {"unique": 0, "three": 0, "saddle-node": 0}
    for B in B_values:
        for A in A_values:
            counts[classify_A_B(float(A), float(B))] += 1
    return counts


def implementation_path(alpha: float, T: int = 40, initial: float = 0.20) -> np.ndarray:
    """An implementation path e_t = initial + (1-initial)*(t/(T-1))**alpha.

    Every admissible alpha > 0 starts at the same initial implementation
    and reaches full implementation at the final transition date.
    """
    if alpha <= 0 or T < 2 or not (0.0 <= initial < 1.0):
        raise ValueError("Require alpha>0, T>=2, and 0<=initial<1.")
    u = np.arange(T, dtype=float) / (T - 1)
    return initial + (1.0 - initial) * u**alpha


def fixed_weakest_link_paths(
    leader_alpha: float = 0.3, laggard_alpha: float = 3.0, T: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return synchronized and fast-leader histories with identical weakest links.

    Path S: both jurisdictions use laggard_alpha.
    Path F: leader uses leader_alpha <= laggard_alpha, laggard unchanged.
    """
    if not 0 < leader_alpha <= laggard_alpha:
        raise ValueError("Require 0 < leader_alpha <= laggard_alpha.")
    laggard = implementation_path(laggard_alpha, T)
    return laggard.copy(), laggard.copy(), implementation_path(leader_alpha, T), laggard.copy()


def final_maturity(leader_alpha: float, laggard_alpha: float, p0: float,
                   par: Params, T: int = 40) -> float:
    """p_T for the exponent pair; alpha_1 <= alpha_2 is the leader convention."""
    if not 0 < leader_alpha <= laggard_alpha:
        raise ValueError("Require 0 < leader_alpha <= laggard_alpha.")
    e1 = implementation_path(leader_alpha, T)
    e2 = implementation_path(laggard_alpha, T)
    return float(simulate(e1, e2, p0, par)["p"][-1])


def critical_leader_exponent(
    laggard_alpha: float, p0: float, p_threshold: float, par: Params,
    T: int = 40, lower: float = 0.20,
) -> float:
    """Unique crossing p_T(alpha_1,alpha_2)=p_threshold, when bracketed.

    Under the sufficient conditions of Proposition 3, p_T rises strictly
    with leader exponent on the admissible interval alpha_1<=alpha_2.
    """
    if not 0 < lower < laggard_alpha:
        raise ValueError("Require 0 < lower < laggard_alpha.")
    f = lambda a: final_maturity(a, laggard_alpha, p0, par, T)-p_threshold
    return bisection_root(f, lower, laggard_alpha, tol=1e-13)


def analytical_dI_de1_fixed_weakest_link(e1: float, e2: float, p: float,
                                         par: Params) -> float:
    """Leader's impact on maturity investment, when e1>e2."""
    if not e1 > e2:
        raise ValueError("Requires a strictly leading jurisdiction 1.")
    xC, xW, xP = shares(e1, e2, p, par)
    return -par.tau*xP/(2.0*par.sigma) * (
        par.thetaP*xC+(par.thetaP-par.thetaW)*xW
    )


def analytical_dF_dp(e1: float, e2: float, p: float, par: Params) -> float:
    """Derivative of the one-period maturity update F with respect to p."""
    xC, xW, xP = shares(e1, e2, p, par)
    return (1.0-par.delta) + par.lam*xP/par.sigma * (
        par.thetaP*xC+(par.thetaP-par.thetaW)*xW
    )


def no_hysteresis_control(par: Params) -> Params:
    """Return a nearby parameterization with B < 4 by lowering lambda."""
    return Params(
        cC=par.cC,
        cW_tilde=par.cW_tilde,
        cP=par.cP,
        phi=par.phi,
        tau=par.tau,
        lam=3.0,
        sigma=par.sigma,
        delta=par.delta,
        thetaW=par.thetaW,
        thetaP=par.thetaP,
    )

def header(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def main() -> None:
    # Transparent illustrative primitive parameterization.
    par = Params()
    p1 = verify_proposition_1(par)

    header("1. Primitive parameter example")
    print(par)
    for key in ["r", "psi0", "psi1", "A", "B"]:
        print(f"{key:>8s} = {p1[key]: .10f}")
    print(f"Number of equilibria = {p1['num_roots']}")
    print(f"Three-equilibrium condition = {p1.get('three_eq_condition', False)}")

    header("2. Equilibria and Proposition 2 stability")
    p2 = verify_proposition_2(par)
    for i, (x, p, gp, stable) in enumerate(zip(p2["x"], p2["p"], p2["Gprime"], p2["stable"]), 1):
        print(f"eq {i}: x={x:.10f}, p={p:.10f}, G'(p)={gp:.10f}, stable={stable}")

    header("3. Lemma 2 analytical-versus-numerical derivative")
    ebar = 0.60
    d = 0.30
    phat = p_hat(ebar, par)
    print(f"p_hat(ebar={ebar}) = {phat:.10f}")
    if np.isfinite(phat) and phat > 0:
        tests = [0.5 * phat, 1.5 * phat]
    else:
        tests = [0.10, 0.50]
    for p in tests:
        e1 = ebar + d / 2.0
        e2 = ebar - d / 2.0
        ana = analytical_dI_dd(e1, e2, p, par)
        num = numerical_dI_dd(ebar, d, p, par)
        print(f"p={p:.10f}: analytical={ana:+.10e}, numerical={num:+.10e}")
        assert abs(ana - num) < 1e-6

    header("4. Leader-versus-laggard comparative statics")
    e1, e2, p = 0.80, 0.40, 0.30
    a1 = analytical_dxW_de1(e1, e2, p, par)
    n1 = numerical_dxW_de1(e1, e2, p, par)
    a2 = analytical_dxW_de2(e1, e2, p, par)
    n2 = numerical_dxW_de2(e1, e2, p, par)
    print(f"dxW/de1 analytical={a1:+.10e}, numerical={n1:+.10e}")
    print(f"dxW/de2 analytical={a2:+.10e}, numerical={n2:+.10e}")
    assert abs(a1 - n1) < 1e-6
    assert abs(a2 - n2) < 1e-6

    header("4a. Proposition 3: fixed-weakest-link derivatives")
    dI_ana = analytical_dI_de1_fixed_weakest_link(e1, e2, p, par)
    h = 1e-6
    dI_num = (maturity_investment(e1+h,e2,p,par)
              - maturity_investment(e1-h,e2,p,par))/(2*h)
    dF_ana = analytical_dF_dp(e1,e2,p,par)
    dF_num = (maturity_next(e1,e2,p+h,par)
              - maturity_next(e1,e2,p-h,par))/(2*h)
    print(f"dI/de_leader analytical={dI_ana:+.10e}, numerical={dI_num:+.10e}")
    print(f"dF/dp analytical={dF_ana:+.10e}, numerical={dF_num:+.10e}")
    assert abs(dI_ana-dI_num)<1e-8 and dI_ana<0
    assert abs(dF_ana-dF_num)<1e-8 and dF_ana>0

    header("5. Basin-boundary verification")
    pL, pM, pH = p2["p"]
    eps = 1e-3
    below = post_implementation_tail(pM - eps, par, periods=2000)
    above = post_implementation_tail(pM + eps, par, periods=2000)
    print(f"pL={pL:.10f}, pM={pM:.10f}, pH={pH:.10f}")
    print(f"start pM-eps -> {below['p'][-1]:.10f}")
    print(f"start pM+eps -> {above['p'][-1]:.10f}")

    header("6. Fixed-weakest-link transition paths and equilibrium selection")
    p0 = pL
    e1_S, e2_S, e1_F, e2_F = fixed_weakest_link_paths(
        leader_alpha=0.3, laggard_alpha=3.0, T=40
    )
    hist_S = simulate(e1_S, e2_S, p0, par)
    hist_F = simulate(e1_F, e2_F, p0, par)
    pT_S = hist_S["p"][-1]
    pT_F = hist_F["p"][-1]
    assert np.allclose(np.minimum(e1_S,e2_S),np.minimum(e1_F,e2_F))
    assert np.all(e1_F >= e1_S)  # Faster leader; laggard held fixed.
    assert np.any(e1_F > e1_S)
    assert np.isclose(e1_S[-1],1.0) and np.isclose(e1_F[-1],1.0)
    tail_S = post_implementation_tail(pT_S, par, periods=2000)
    tail_F = post_implementation_tail(pT_F, par, periods=2000)
    print(f"p0={p0:.10f}; pM={pM:.10f}")
    print(f"synchronized alpha=(3,3): pT={pT_S:.10f}, long-run={tail_S['p'][-1]:.10f}")
    print(f"fast leader alpha=(0.3,3): pT={pT_F:.10f}, long-run={tail_F['p'][-1]:.10f}")
    assert abs(pT_S-0.5180261096) < 1e-8
    assert abs(pT_F-0.4632149059) < 1e-8
    assert pT_F < pM < pT_S
    assert abs(tail_S["p"][-1]-pH) < 1e-6
    assert abs(tail_F["p"][-1]-pL) < 1e-6

    header("6a. Unique crossing and two-dimensional map checkpoints")
    alpha_star = critical_leader_exponent(3.0,p0,pM,par)
    print(f"critical alpha_1 for alpha_2=3: {alpha_star:.10f}")
    assert abs(alpha_star-1.0330745622) < 1e-8
    sample_a1 = [0.3,0.5,1.0,1.5,2.0,3.0]
    sample_pT = [final_maturity(a1,3.0,p0,par) for a1 in sample_a1]
    for a1,pT in zip(sample_a1,sample_pT):
        print(f"leader_alpha={a1:.1f}, laggard_alpha=3.0: pT={pT:.10f}")
    assert all(left<right for left,right in zip(sample_pT,sample_pT[1:]))
    assert sample_pT[2]<pM<sample_pT[3]
    for laggard_alpha in (2.8,3.0,3.2):
        for leader_alpha in (0.2,0.3,0.5):
            assert final_maturity(leader_alpha,laggard_alpha,p0,par)<pM
        assert final_maturity(laggard_alpha,laggard_alpha,p0,par)>pM
    print("Fixed-laggard ranking and local 2-D map checkpoints passed.")

    header("7. No-hysteresis control (same updated paths)")
    par_u = no_hysteresis_control(par)
    u1 = verify_proposition_1(par_u)
    print(f"control B = {u1['B']:.10f}")
    print(f"control number of equilibria = {u1['num_roots']}")
    assert u1["B"] < 4.0
    assert u1["num_roots"] == 1
    x_u = u1["roots"][0]
    p_u = maturity_from_x(x_u, par_u)
    hist_S_u = simulate(e1_S,e2_S,p0,par_u)
    hist_F_u = simulate(e1_F,e2_F,p0,par_u)
    tail_S_u = post_implementation_tail(hist_S_u["p"][-1],par_u,periods=2000)
    tail_F_u = post_implementation_tail(hist_F_u["p"][-1],par_u,periods=2000)
    print(f"unique steady-state p = {p_u:.10f}")
    print(f"synchronized long-run p = {tail_S_u['p'][-1]:.10f}")
    print(f"fast-leader long-run p = {tail_F_u['p'][-1]:.10f}")
    assert abs(tail_S_u["p"][-1] - p_u) < 1e-6
    assert abs(tail_F_u["p"][-1] - p_u) < 1e-6

    header("8. Broad reduced-form parameter-region scan")
    A_values = np.linspace(-8.0, 3.0, 221)
    B_values = np.linspace(0.0, 12.0, 241)
    counts = scan_reduced_parameter_region(A_values, B_values)
    total = sum(counts.values())
    for key, val in counts.items():
        print(f"{key:>12s}: {val:6d} ({100.0*val/total:6.2f}%)")
    assert counts["unique"] > 0
    assert counts["three"] > 0

    header("9. Automated consistency checks")
    assert max(abs(r) for r in p1["residuals"]) < 1e-8
    assert abs(sum(shares(0.70, 0.40, 0.30, par)) - 1.0) < 1e-12
    assert p2["stable"] == [True, False, True]
    assert abs(below["p"][-1] - pL) < 1e-6
    assert abs(above["p"][-1] - pH) < 1e-6
    print("All automated checks passed.")


if __name__ == "__main__":
    main()