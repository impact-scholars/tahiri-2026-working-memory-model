"""
==========
Run the working memory model for all psychiatric conditions,
save results to CSV, and generate comparison figures.

Usage
-----
    python run_all_conditions.py                  # all conditions, 200 ms delay
    python run_all_conditions.py --delay 500      # 500 ms delay
    python run_all_conditions.py --condition control --condition -NMDA
    python run_all_conditions.py --no-plot        # suppress individual plots

Author : Hamza
Date   : April 2026
"""

import argparse
import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

from brian2 import ms, Hz, nS, mV, nF, kHz, mmole, pA, second
from Working_Memory_Model import run_wm_combined, DEFAULT_CFG, COND_PARAMS

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="WM model batch runner")
    p.add_argument("--delay",    type=float, default=500,
                   help="Delay period in ms (default: 500)")
    p.add_argument("--N",        type=int,   default=2000,
                   help="Total neuron count (default: 2000)")
    p.add_argument("--condition", action="append", dest="conditions",
                   help="Run only these conditions (repeatable)")
    p.add_argument("--no-plot",  action="store_true",
                   help="Suppress individual trial plots")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


# ── Base configuration ────────────────────────────────────────────────────────
def make_base_cfg(args):
    cfg = dict(DEFAULT_CFG)
    cfg.update(
        baseline=200 * ms,
        cue_dur=500 * ms,
        delay=args.delay * ms,
        N_total=args.N,
        seed=args.seed,
        # Standard distractor: pool S2 stimulated 300 ms into delay
        distractors=[
            dict(t_on=900 * ms, t_off=1100 * ms,
                 rate=200 * Hz, pool=1, mode="one_to_one")
        ],
        plot=False,
        codegen_target="numpy",
    )
    return cfg


# ── Comparison plot ───────────────────────────────────────────────────────────
def plot_comparison(all_monitors, all_cfgs, all_metrics, conditions):
    """One row per condition: S1 rate, S2 rate, I rate."""
    n   = len(conditions)
    fig = plt.figure(figsize=(18, 3.5 * n))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.35)

    colour = {"S1": "#1f77b4", "S2": "#ff7f0e", "I": "#2ca02c"}

    for row, cond in enumerate(conditions):
        cfg_used = all_cfgs[cond]
        mon      = all_monitors[cond]
        met      = all_metrics[cond]
        cue_on   = cfg_used["cue_on"] / ms
        cue_off  = cfg_used["cue_off"] / ms
        span_kw  = dict(alpha=0.15, color="gold")
        dis      = cfg_used.get("distractors", [])

        for col, (key, lbl) in enumerate([("R1", "S1"), ("R2", "S2"), ("RI", "I")]):
            ax  = fig.add_subplot(gs[row, col])
            R   = mon[key]
            t   = R.t / ms
            r   = R.smooth_rate(window="flat", width=50 * ms) / Hz
            ax.plot(t, r, lw=1.2, color=colour[lbl])
            ax.axvspan(cue_on, cue_off, **span_kw)
            for ev in dis:
                ax.axvspan(ev["t_on"] / ms, ev["t_off"] / ms,
                           alpha=0.10, color="red")
            ax.set_title(f"{cond} | {lbl}", fontsize=8)
            ax.set_xlabel("Time (ms)", fontsize=7)
            ax.set_ylabel("Rate (Hz)", fontsize=7)
            ax.tick_params(labelsize=7)

        # annotate persistence
        met_str = (f"persist={met['persistent']}  "
                   f"λ={met['lambda_delay']:.2f}  "
                   f"S1_delay={met['r1_delay']:.1f} Hz")
        fig.text(0.5, gs[row, :].get_position(fig).y1 + 0.005,
                 met_str, ha="center", fontsize=7, color="grey")

    fig.suptitle("Working Memory Model — Condition Comparison", fontsize=13, y=1.01)
    plt.savefig("figures/comparison_all_conditions.png", dpi=150, bbox_inches="tight")
    print("Saved → figures/comparison_all_conditions.png")
    plt.show()


def plot_metrics_bar(df):
    """Bar chart of key metrics across conditions."""
    metrics_to_plot = ["r1_delay", "r2_delay", "frac_above_thr",
                       "selective_frac", "lambda_delay"]
    labels = {
        "r1_delay":       "S1 delay rate (Hz)",
        "r2_delay":       "S2 delay rate (Hz)",
        "frac_above_thr": "Frac. above thr.",
        "selective_frac": "Selectivity frac.",
        "lambda_delay":   "λ decay (lower=better)",
    }

    fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(16, 4))
    conds    = df["condition"].tolist()
    x        = np.arange(len(conds))
    colours  = ["#2ca02c" if "control" in c else
                "#d62728" if "NMDA" in c and "-" in c else
                "#1f77b4"
                for c in conds]

    for ax, met in zip(axs, metrics_to_plot):
        vals = df[met].fillna(0).tolist()
        bars = ax.bar(x, vals, color=colours)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=7)
        ax.set_title(labels[met], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        # mark persistent
        for xi, (v, p) in enumerate(zip(vals, df["persistent"].tolist())):
            if p:
                ax.text(xi, v + 0.2, "✓", ha="center", fontsize=8, color="green")

    plt.suptitle("Metrics Across Conditions", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/metrics_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved → figures/metrics_comparison.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args       = parse_args()
    base_cfg   = make_base_cfg(args)
    conditions = args.conditions or list(COND_PARAMS.keys())

    all_monitors = {}
    all_cfgs     = {}
    all_metrics  = {}
    rows         = []

    for cond in conditions:
        print(f"\n{'='*72}")
        print(f"  RUNNING:  {cond}")
        print(f"{'='*72}")

        cfg = dict(base_cfg)
        cfg["condition"] = cond

        t0 = time.time()
        cfg_used, monitors, metrics = run_wm_combined(
            cfg, plot=(not args.no_plot), report="text"
        )
        elapsed = time.time() - t0

        all_monitors[cond] = monitors
        all_cfgs[cond]     = cfg_used
        all_metrics[cond]  = metrics

        row = {"condition": cond, "elapsed_s": round(elapsed, 1)}
        row.update(metrics)
        rows.append(row)

        print(f"  Done in {elapsed:.1f} s")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = "results/all_conditions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print(df[["condition", "r1_delay", "r2_delay",
              "persistent", "frac_above_thr", "lambda_delay"]].to_string(index=False))

    # ── Comparison figures ────────────────────────────────────────────────────
    print("\nGenerating comparison figures …")
    plot_comparison(all_monitors, all_cfgs, all_metrics, conditions)
    plot_metrics_bar(df)

    print("\n All done.")


if __name__ == "__main__":
    main()
