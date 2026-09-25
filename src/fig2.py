#!/usr/bin/env python3
"""
fig2.py

Figure 2 for:
"Implementation Gaps, Channel Substitution, and Hysteresis in Crypto-Asset Regulation"

The figure shows the post-implementation fixed-point map

    x = 1 / (1 + exp[-(A + Bx)])

together with the 45-degree line. The three intersections are the
low-, middle-, and high-P2P equilibria characterized in Proposition 1.

The parameter values and equilibrium calculations are imported from
verify_model.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from verify_model import (
    Params,
    reduced_objects,
    verify_proposition_2,
)


def main() -> None:
    outdir = Path(__file__).resolve().parent

    par = Params()
    obj = reduced_objects(par)
    stability = verify_proposition_2(par)

    if len(stability["x"]) != 3:
        raise RuntimeError("The illustrative parameterization must have three equilibria.")

    A = obj["A"]
    B = obj["B"]
    xL, xM, xH = stability["x"]

    x_grid = np.linspace(0.0, 1.0, 1001)
    z = A + B * x_grid
    fixed_map = np.where(
        z >= 0.0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    ax.plot(x_grid, x_grid, label=r"$45^\circ$ line")
    ax.plot(
        x_grid,
        fixed_map,
        label=r"$[1+\exp\{-(A+Bx)\}]^{-1}$",
    )

    eq_x = np.array([xL, xM, xH])
    ax.scatter(eq_x, eq_x)

    ax.annotate(
        r"$x_L$ (stable)",
        xy=(xL, xL),
        xytext=(12, 12),
        textcoords="offset points",
    )
    ax.annotate(
        r"$x_M$ (unstable)",
        xy=(xM, xM),
        xytext=(12, -22),
        textcoords="offset points",
    )
    ax.annotate(
        r"$x_H$ (stable)",
        xy=(xH, xH),
        xytext=(-105, -28),
        textcoords="offset points",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Current P2P share $x$")
    ax.set_ylabel(r"Implied P2P share")
    ax.set_title("Post-implementation fixed points")
    ax.legend(frameon=False)

    fig.tight_layout()

    pdf_path = outdir / "Fig2_fixed_points.pdf"
    png_path = outdir / "Fig2_fixed_points.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"A = {A:.10f}")
    print(f"B = {B:.10f}")
    print(f"x_L = {xL:.10f}")
    print(f"x_M = {xM:.10f}")
    print(f"x_H = {xH:.10f}")


if __name__ == "__main__":
    main()