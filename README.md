# Implementation Gaps, Channel Substitution, and Hysteresis in Crypto-Asset Regulation

Replication code for the second revised version of the *Digital Finance* manuscript.
The model concerns a bilateral cross-border crypto-asset corridor in which users choose among compliant VASPs, weak/non-compliant VASPs, and P2P transactions. P2P maturity accumulates as a stock.

## Files

- `src/verify_model.py` — verifies reduced-form equilibrium conditions, local stability, the basin boundary, the transitional learning mechanism, leader/laggard share derivatives, **Proposition 3's fixed-weakest-link comparative dynamics**, the **critical implementation-speed exponent**, and the **unique-equilibrium benchmark**. The historical A/B paths from the first revision are no longer used.
- `src/fig1.py` — creates revised Figure 1, *Implementation speeds and long-run equilibrium selection*. Files: `Fig1_implementation_speeds.pdf` and `Fig1_implementation_speeds.png`.
- `src/fig2.py` — creates Figure 2, *Post-implementation fixed points*. Files: `Fig2_fixed_points.pdf` and `Fig2_fixed_points.png`.
- `requirements.txt` — minimal Python dependencies.

## Run

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
python src/verify_model.py
python src/fig1.py
python src/fig2.py
```

The figure files are saved in the `src/` directory.

## Revised Figure 1 and Appendix A

Both jurisdictions follow

\[
e_{j,t}=0.20+0.80\left(\frac{t}{T-1}\right)^{\alpha_j},\qquad
T=40,\quad t=0,\ldots,T-1,\quad 0<\alpha_1\le\alpha_2.
\]

The laggard's exponent is fixed at \(\alpha_2=3\) in the controlled comparison, so the weakest-link implementation path is identical throughout the transition. Initial maturity is \(p_0=p_L\).

| Implementation path | \((\alpha_1,\alpha_2)\) | Inherited \(p_T\) | Long-run equilibrium |
|---|---|---:|---|
| Faster leader | \((0.3,3)\) | 0.4632149059 | Low-P2P |
| Synchronized | \((3,3)\) | 0.5180261096 | High-P2P |

The post-implementation basin threshold is \(p_M=0.4889371594\). The critical leader exponent at \(\alpha_2=3\) is \(\alpha_1^*=1.0330745622\). The Figure 1 script generates the two-dimensional equilibrium-selection map and independently cross-checks vectorized calculations against the scalar model.

The single-equilibrium control sets \(\lambda=3\), yielding the same long-run equilibrium under both implementation paths.

## Scope

These calculations are theoretical illustrations, **not empirical calibration**. No external dataset is required. The source code implements the model's equations and verifies the numerical results reported in the manuscript and Appendix A.

## License

See `LICENSE`.