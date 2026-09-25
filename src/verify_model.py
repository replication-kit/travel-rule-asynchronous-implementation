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
5. Lemma 2: when implementation asymmetry increases P2P maturity investment.
6. Leader-versus-laggard comparative statics.
7. A transition-path example in which two implementation histories converge
   to the same terminal regulatory state but different long-run equilibria.

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


def illustrative_transition_paths(T: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two monotone implementation histories with the same initial and terminal
    regulatory states.  They are chosen only to illustrate Proposition 3.

    History A: synchronized implementation.
    History B: strong leader-laggard asymmetry followed by catch-up.
    """
    t = np.arange(T, dtype=float)
    u = t / (T - 1)

    # History A: both jurisdictions move together.
    sync = 0.20 + 0.80 * u**2.0
    e1_A = sync.copy()
    e2_A = sync.copy()

    # History B: the leader reaches full implementation relatively early,
    # while the laggard remains weak for longer and catches up near the end.
    leader = np.minimum(1.0, 0.20 + 0.80 * t / max(0.50 * (T - 1), 1.0))
    laggard = 0.20 + 0.80 * u**10.0
    e1_B = leader
    e2_B = laggard

    # Force an identical terminal state.
    e1_A[-1] = e2_A[-1] = 1.0
    e1_B[-1] = e2_B[-1] = 1.0

    return e1_A, e2_A, e1_B, e2_B


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

    header("5. Basin-boundary verification")
    pL, pM, pH = p2["p"]
    eps = 1e-3
    below = post_implementation_tail(pM - eps, par, periods=2000)
    above = post_implementation_tail(pM + eps, par, periods=2000)
    print(f"pL={pL:.10f}, pM={pM:.10f}, pH={pH:.10f}")
    print(f"start pM-eps -> {below['p'][-1]:.10f}")
    print(f"start pM+eps -> {above['p'][-1]:.10f}")

    header("6. Transition-path hysteresis example")
    p0 = pL
    e1_A, e2_A, e1_B, e2_B = illustrative_transition_paths(T=40)
    histA = simulate(e1_A, e2_A, p0, par)
    histB = simulate(e1_B, e2_B, p0, par)
    pTA = histA["p"][-1]
    pTB = histB["p"][-1]
    tailA = post_implementation_tail(pTA, par, periods=2000)
    tailB = post_implementation_tail(pTB, par, periods=2000)
    print(f"initial p0={p0:.10f}, threshold pM={pM:.10f}")
    print(f"history A: pT={pTA:.10f}, long-run={tailA['p'][-1]:.10f}")
    print(f"history B: pT={pTB:.10f}, long-run={tailB['p'][-1]:.10f}")
    opposite = (pTA < pM < pTB) or (pTB < pM < pTA)
    print(f"opposite sides of basin boundary = {opposite}")
    assert opposite
    assert abs(tailA["p"][-1] - pL) < 1e-6
    assert abs(tailB["p"][-1] - pH) < 1e-6

    header("7. No-hysteresis control (unique-equilibrium region)")
    par_u = no_hysteresis_control(par)
    u1 = verify_proposition_1(par_u)
    print(f"control B = {u1['B']:.10f}")
    print(f"control number of equilibria = {u1['num_roots']}")
    assert u1["B"] < 4.0
    assert u1["num_roots"] == 1
    x_u = u1["roots"][0]
    p_u = maturity_from_x(x_u, par_u)
    histA_u = simulate(e1_A, e2_A, p0, par_u)
    histB_u = simulate(e1_B, e2_B, p0, par_u)
    tailA_u = post_implementation_tail(histA_u["p"][-1], par_u, periods=2000)
    tailB_u = post_implementation_tail(histB_u["p"][-1], par_u, periods=2000)
    print(f"unique steady-state p = {p_u:.10f}")
    print(f"history A long-run p = {tailA_u['p'][-1]:.10f}")
    print(f"history B long-run p = {tailB_u['p'][-1]:.10f}")
    assert abs(tailA_u["p"][-1] - p_u) < 1e-6
    assert abs(tailB_u["p"][-1] - p_u) < 1e-6

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