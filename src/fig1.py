#!/usr/bin/env python3
"""
fig1.py

Revised Figure 1 for:
"Implementation Gaps, Channel Substitution, and Hysteresis in Crypto-Asset Regulation"

Key revision:
- The unstable threshold p_M is a post-implementation basin boundary.
- It is therefore shown only from T onward, after both jurisdictions have
  reached the common terminal regulatory state e1=e2=1.
- A vertical line at T separates the transition period from the
  post-implementation period.
- p_T^A and p_T^B are marked explicitly at T.

Place this file in the same directory as verify_model.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from verify_model import (
    Params,
    illustrative_transition_paths,
    post_implementation_tail,
    simulate,
    verify_proposition_2,
)


def main() -> None:
    outdir = Path(__file__).resolve().parent

    par = Params()
    stability = verify_proposition_2(par)

    if len(stability["p"]) != 3:
        raise RuntimeError(
            "The illustrative parameterization must generate three equilibria."
        )

    pL, pM, pH = stability["p"]

    # Both histories start from the same maturity.
    p0 = pL

    # Forty transition periods. After these periods, both jurisdictions
    # remain at e1=e2=1.
    T_transition = 40
    e1_A, e2_A, e1_B, e2_B = illustrative_transition_paths(T=T_transition)

    hist_A = simulate(e1_A, e2_A, p0, par)
    hist_B = simulate(e1_B, e2_B, p0, par)

    # p_T is the inherited maturity after the transition.
    pT_A = hist_A["p"][-1]
    pT_B = hist_B["p"][-1]

    if not (pT_A < pM < pT_B):
        raise RuntimeError(
            "The illustrative histories must leave p_T on opposite sides of p_M."
        )

    # Common post-implementation dynamics.
    post_periods = 250
    tail_A = post_implementation_tail(pT_A, par, periods=post_periods)
    tail_B = post_implementation_tail(pT_B, par, periods=post_periods)

    # hist_*["p"] runs from p_0 through p_T.
    # tail_* starts again from p_T, so omit the repeated initial observation.
    p_A = np.concatenate([hist_A["p"], tail_A["p"][1:]])
    p_B = np.concatenate([hist_B["p"], tail_B["p"][1:]])

    time = np.arange(len(p_A))
    T = T_transition

    fig, ax = plt.subplots(figsize=(7.6, 4.9))

    ax.plot(time, p_A, label="History A: synchronized implementation")
    ax.plot(time, p_B, label="History B: asynchronous implementation")

    # The basin boundary p_M is defined only for the common
    # post-implementation system, so draw it only from T onward.
    post_time = time[T:]
    ax.plot(
        post_time,
        np.full(post_time.shape, pM, dtype=float),
        linestyle="--",
        label=r"Post-implementation threshold $p_M$",
    )

    # Mark the date at which the transition ends.
    ax.axvline(
        T,
        linestyle=":",
        label=r"Full implementation ($T=40$)",
    )

    # Mark inherited maturity at T.
    ax.scatter([T, T], [pT_A, pT_B], zorder=3)

    ax.annotate(
        r"$p_T^A$",
        xy=(T, pT_A),
        xytext=(8, -18),
        textcoords="offset points",
    )
    ax.annotate(
        r"$p_T^B$",
        xy=(T, pT_B),
        xytext=(8, 10),
        textcoords="offset points",
    )

    # Label the long-run stable equilibria near the right edge.
    ax.annotate(
        r"$p_L$",
        xy=(time[-1], pL),
        xytext=(-30, 8),
        textcoords="offset points",
    )
    ax.annotate(
        r"$p_H$",
        xy=(time[-1], pH),
        xytext=(-30, -15),
        textcoords="offset points",
    )

    # Period labels make clear that p_M is not a transition-period threshold.
    ymax = max(float(np.max(p_A)), float(np.max(p_B)), pH)
    ax.text(
        0.5 * T,
        0.98 * ymax,
        "Transition",
        ha="center",
        va="top",
    )
    ax.text(
        T + 0.5 * (time[-1] - T),
        0.98 * ymax,
        "Post-implementation",
        ha="center",
        va="top",
    )

    ax.set_xlabel("Period")
    ax.set_ylabel(r"P2P maturity $p_t$")
    ax.set_title("Transition histories and long-run equilibrium selection")
    ax.legend(frameon=False)

    fig.tight_layout()

    pdf_path = outdir / "Fig1_time_paths_v2.pdf"
    png_path = outdir / "Fig1_time_paths_v2.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"p_L = {pL:.10f}")
    print(f"p_M = {pM:.10f}")
    print(f"p_H = {pH:.10f}")
    print(f"History A: p_T = {pT_A:.10f}")
    print(f"History B: p_T = {pT_B:.10f}")
    print(f"History A long-run p = {tail_A['p'][-1]:.10f}")
    print(f"History B long-run p = {tail_B['p'][-1]:.10f}")


if __name__ == "__main__":
    main()