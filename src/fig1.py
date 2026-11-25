#!/usr/bin/env python3
"""Figure 1: implementation speeds and long-run equilibrium selection.

Reproduces the revised Digital Finance numerical illustration.
Model and parameter definitions are imported from the paper's verify_model.py.

Usage:
    Place this script beside verify_model.py (or set PYTHONPATH accordingly).
    python fig1.py

Outputs:
    Fig1_implementation_speeds.pdf  (vector, journal-ready)
    Fig1_implementation_speeds.png  (600 dpi)

The figure compares only admissible paths alpha_1 <= alpha_2, with
    e_{j,t} = 0.2 + 0.8 * (t/(T-1)) ** alpha_j, t=0,...,T-1.
All paths start at 0.2 and end at 1, and share p_0 = p_L.
The division into high/low basins uses the unstable *post-implementation*
maturity threshold p_M, not a changing transition-period equilibrium.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verify_model import Params, verify_proposition_2, simulate

T = 40
E0 = 0.20
U = np.arange(T, dtype=float) / (T-1)
PAR = Params()
P_L, P_M, P_H = verify_proposition_2(PAR)["p"]
OUT = Path(__file__).resolve().parent


def path(alpha: float) -> np.ndarray:
    return E0 + (1-E0) * U ** alpha


def final_maturity(a_leader: float, a_laggard: float) -> float:
    return float(simulate(path(a_leader), path(a_laggard), P_L, PAR)["p"][-1])


def grid_final_maturity(a_leader: np.ndarray, a_laggard: np.ndarray) -> np.ndarray:
    """Vectorized simulation, same logit shares and timing as verify_model.py."""
    aa, bb = np.broadcast_arrays(a_leader, a_laggard)
    p = np.full(aa.shape, P_L, dtype=float)
    for u in U:
        e1 = E0 + (1-E0) * u**aa
        e2 = E0 + (1-E0) * u**bb
        # Stable multinomial-logit computation.
        zC = np.full(aa.shape, -PAR.cC/PAR.sigma)
        zW = -(PAR.cW_tilde+PAR.phi*np.minimum(e1,e2))/PAR.sigma
        zP = -(PAR.cP + PAR.tau*(e1+e2)/2 - PAR.lam*p)/PAR.sigma
        zmax = np.maximum.reduce((zC,zW,zP))
        wC, wW, wP = np.exp(zC-zmax),np.exp(zW-zmax),np.exp(zP-zmax)
        den = wC+wW+wP
        p = (1-PAR.delta)*p + (PAR.thetaW*wW+PAR.thetaP*wP)/den
    return p


def bisect_critical_alpha(laggard_alpha: float = 3.0) -> float:
    left, right = .25, laggard_alpha
    assert final_maturity(left,laggard_alpha) < P_M
    assert final_maturity(right,laggard_alpha) > P_M
    for _ in range(80):
        middle = (left+right)/2
        if final_maturity(middle,laggard_alpha) < P_M:
            left = middle
        else:
            right = middle
    return (left+right)/2


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    # Verify the previously selected benchmark comparison and threshold.
    p_fast = final_maturity(.3,3.)
    p_sync = final_maturity(3.,3.)
    star = bisect_critical_alpha()
    assert abs(P_M-.4889371594) < 1e-8
    assert abs(p_fast-.463215) < 2e-6, p_fast
    assert abs(p_sync-.518026) < 2e-6, p_sync
    assert abs(star-1.0330745622) < 1e-6, star
    assert p_fast < P_M < p_sync

    a1 = np.linspace(.20, 6.0, 520)
    a2 = np.linspace(.20, 10.0, 620)
    A1,A2 = np.meshgrid(a1,a2)
    maturity = grid_final_maturity(A1,A2)
    valid = A1<=A2
    diff = np.ma.masked_where(~valid,maturity-P_M)

    # Cross-check the vectorized grid against the original scalar model.
    for aa,bb in ((.3,3.),(1.,3.),(1.5,3.),(3.,3.),(2.,8.)):
        vec = grid_final_maturity(np.array(aa),np.array(bb)).item()
        scalar = final_maturity(aa,bb)
        assert abs(vec-scalar) < 1e-12, (aa,bb,vec,scalar)

    fig, ax = plt.subplots(figsize=(6.65,4.83), dpi=120)
    fig.subplots_adjust(left=.119,right=.98,top=.968,bottom=.145)

    # Publication grayscale. White denotes inadmissible combinations.
    ax.contourf(A1,A2,diff,levels=[-2.0,0.0,2.0],
                colors=["#f1f1f1", "#bcbcbc"],
                antialiased=False,zorder=1)

    # Equilibrium-selection boundary p_T = p_M.
    contour = ax.contour(A1,A2,diff,levels=[0.],colors=["black"],
                         linewidths=2.15,zorder=4)

    # Dotted diagonal is synchronized implementation, alpha_1 = alpha_2.
    ax.plot([.2,6],[.2,6],color="#424242",lw=1.25,
            dashes=(4.0,2.4),zorder=3)
    # alpha_2 = 3 slice; comparison holds laggard's path fixed.
    ax.plot([.2,3.],[3.,3.],color="#595959",lw=1.08,
            dashes=(6.,2.,1.2,2.),zorder=3)

    # Critical leader speed along that controlled slice.
    ax.scatter([star],[3.],s=38,marker='D',facecolors='white',
               edgecolors='black',linewidths=1.15,zorder=7)
    ax.annotate(r"$\alpha_1^{*}=1.033$",xy=(star,3.),xycoords='data',
                xytext=(1.57,4.11),textcoords='data',
                fontsize=9.1,ha='left',va='center',
                arrowprops=dict(arrowstyle='-',color='black',lw=.86,
                                shrinkA=5,shrinkB=6,connectionstyle='angle3,angleA=0,angleB=-90'),
                zorder=7)

    # Two selected examples at fixed weakest-link path (alpha_2 = 3).
    ax.scatter([.3],[3.],s=46,marker='o',facecolors='black',
               edgecolors='white',linewidths=.7,zorder=8)
    ax.scatter([3.],[3.],s=46,marker='s',facecolors='white',
               edgecolors='black',linewidths=1.35,zorder=8)
    ax.annotate("Fast leader",xy=(.3,3.),xytext=(.42,2.45),
                fontsize=8.65,ha='left',va='center',
                arrowprops=dict(arrowstyle='-',color='#444444',lw=.65,
                                shrinkA=3,shrinkB=4),zorder=7)
    ax.annotate("Synchronized",xy=(3.,3.),xytext=(3.18,2.66),
                fontsize=8.65,ha='left',va='center',
                arrowprops=dict(arrowstyle='-',color='#444444',lw=.65,
                                shrinkA=3,shrinkB=4),zorder=7)

    # Distinct, short region labels. No colorbar: classes are not magnitudes.
    ax.text(.50,1.85,"Low-P2P\nequilibrium",ha='left',va='center',fontsize=9.1,
            linespacing=1.23,zorder=5)
    ax.text(3.0,7.48,"High-P2P\nequilibrium",ha='center',va='center',fontsize=9.1,
            linespacing=1.23,zorder=5)
    ax.text(3.95,5.1,"$\\alpha_1=\\alpha_2$",fontsize=8.5,rotation=40,
            color="#424242",zorder=5)

    ax.set_xlim(.2,6.)
    ax.set_ylim(.2,10.)
    ax.set_xticks([.5,1,2,3,4,5,6])
    ax.set_yticks([1,2,3,4,6,8,10])
    ax.set_xlabel(r"Leader path exponent $\alpha_1$ (smaller = faster)",labelpad=6)
    ax.set_ylabel(r"Laggard path exponent $\alpha_2$ (smaller = faster)",labelpad=6)
    ax.tick_params(direction='out',length=3.5,width=.8)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)

    pdf = OUT / "Fig1_implementation_speeds.pdf"
    png = OUT / "Fig1_implementation_speeds.png"
    fig.savefig(pdf,format="pdf",metadata={
        "Title":"Implementation speeds and long-run equilibrium selection",
        "Creator":"Matplotlib; reproducible numerical model",
    })
    fig.savefig(png,format="png",dpi=600)
    plt.close(fig)
    print(f"p_L = {P_L:.10f}; p_M = {P_M:.10f}; p_H = {P_H:.10f}")
    print(f"alpha_1*= {star:.10f}; p_T(0.3,3) = {p_fast:.10f}; p_T(3,3) = {p_sync:.10f}")
    print(f"Wrote {pdf} and {png}")

if __name__=='__main__':
    main()