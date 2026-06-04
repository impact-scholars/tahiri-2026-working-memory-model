# Run_Grid_Trials.py
"""
Grid search over (nmda_scale, gaba_scale, cue_rate, Jp,
                  distractor_rate, seed).

Key additions vs previous version
-----------------------------------
1. `distractor_rates` axis  – includes 0 Hz (no distractor).
2. `seeds` list             – each combo is replicated over N seeds.
3. Each row in the CSV carries `distractor_rate` and `seed` columns so
   you can later aggregate / compare across seeds or distractor strengths.
4. All other behaviour (incremental CSV writing, heatmaps, CLI flags)
   is preserved and extended.
"""

import argparse
import os
import time
import warnings
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from brian2 import ms, Hz, nS
from Working_Memory_Model import run_wm_combined, DEFAULT_CFG, COND_PARAMS


# ══════════════════════════════════════════════════════════════════════════════
# GRID DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

FULL_GRID = dict(
    nmda_scale       = [1.10, 1.00, 0.90],
    gaba_scale       = [1.00, 0.90],
    cue_rate         = [200],
    Jp               = [1.84, 1.85, 1.86],
    # 0 Hz = no distractor trial; other values = distractor strengths
    distractor_rates = [0, 200],
    seeds            = [42, 123, 7, 2024, 99, 1, 54, 65],
)

QUICK_GRID = dict(
    nmda_scale       = [0.95, 1.00, 1.10],
    gaba_scale       = [1.00],
    cue_rate         = [200],
    Jp               = [1.84],
    distractor_rates = [0, 200],   # no-distractor vs strong distractor
    seeds            = [42, 123],
)

# ── Explicit point list ───────────────────────────────────────────────────────
# Each entry is (nmda_scale, gaba_scale, Jp, cue_rate).
# Add / remove rows freely; the runner will cross these with
# distractor_rates and seeds defined just below.

POINTS = [
    # nmda  gaba   Jp     cue_rate
    (0.8,  0.4,  1.88,  200),
    (0.9,  0.3,  1.88,  200),
    (1.0,  0.3,  1.88,  200),
    (1.0,  0.6,  1.88,  200),
    (1.1,  0.4,  1.88,  200),
    (1.1,  0.6,  1.88,  200),

    (0.8,  0.4,  1.75,  200),
    (0.9,  0.3,  1.75,  200),
    (1.0,  0.3,  1.75,  200),
    (1.0,  0.6,  1.75,  200),
    (1.1,  0.4,  1.75,  200),
    (1.1,  0.6,  1.75,  200),
]

POINTS_GRID = dict(
    distractor_rates = [0, 200],
    seeds            = [42, 123, 7, 2024, 99, 1, 54, 65],
)


def points_to_combos(points, distractor_rates, seeds):
    """
    Expand POINTS × distractor_rates × seeds into a flat list of
    (nmda_scale, gaba_scale, cue_rate, Jp, distractor_rate, seed) tuples.
    Matches the tuple order used by run_grid().
    """
    combos = []
    for (nmda_s, gaba_s, jp, cue_r) in points:
        for dr in distractor_rates:
            for seed in seeds:
                combos.append((nmda_s, gaba_s, cue_r, jp, dr, seed))
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def make_trial_cfg(base_cfg, nmda_s, gaba_s, cue_r, jp, distractor_rate, seed):
    """
    Build a full run_wm_combined config for one grid point.

    Parameters
    ----------
    distractor_rate : float  Hz (0 = no distractor)
    seed            : int
    """
    # Register the condition on-the-fly so run_wm_combined can look it up
    tag = f"g_n{nmda_s:.3f}_g{gaba_s:.3f}"
    COND_PARAMS[tag] = dict(nmda_scale=nmda_s, gaba_scale=gaba_s)

    cfg = dict(base_cfg)          # shallow copy; timing values are Brian2 Quantities
    cfg["condition"] = tag
    cfg["cue_rate"]  = cue_r * Hz
    cfg["Jp"]        = jp
    cfg["seed"]      = seed
    cfg["plot"]      = False      # suppress plots during batch runs

    # ── Distractor ────────────────────────────────────────────────────────────
    if distractor_rate > 0:
        cfg["distractors"] = [
            dict(
                t_on=800 * ms,
                t_off=1100 * ms,
                rate=distractor_rate * Hz,
                pool=1,
                mode="one_to_one",
            )
        ]
    else:
        # Explicitly empty list = no-distractor (pure persistence trial)
        cfg["distractors"] = []

    return cfg, tag


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def run_one(nmda_s, gaba_s, cue_r, jp, distractor_rate, seed, base_cfg):
    """
    Run one simulation and return a flat metrics dict with provenance columns.
    Catches exceptions and stores the error message rather than crashing.
    """
    cfg, tag = make_trial_cfg(base_cfg, nmda_s, gaba_s, cue_r, jp,
                              distractor_rate, seed)

    # Provenance columns (always present in the row)
    row = dict(
        nmda_scale      = nmda_s,
        gaba_scale      = gaba_s,
        cue_rate        = cue_r,
        Jp              = jp,
        distractor_rate = distractor_rate,
        seed            = seed,
    )

    try:
        _cfg_used, _monitors, metrics = run_wm_combined(cfg, plot=False, report=None)
        row.update(metrics)
    except Exception as exc:
        row["error"] = str(exc)

    return row


# ══════════════════════════════════════════════════════════════════════════════
# INCREMENTAL CSV WRITER
# ══════════════════════════════════════════════════════════════════════════════

def append_row_to_csv(row, csv_path):
    """Append one result row to CSV; write header only when file is new."""
    row_df     = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL GRID RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_grid(grid_or_combos, base_cfg, csv_path="results/grid_results.csv",
             resume=False):
    """
    Iterate over every (nmda, gaba, cue, Jp, distractor_rate, seed) combo.

    Parameters
    ----------
    grid_or_combos : dict  – classic grid dict  (keys: nmda_scale, gaba_scale,
                             cue_rate, Jp, distractor_rates, seeds)
                   | list  – explicit list of
                             (nmda_s, gaba_s, cue_r, jp, dr, seed) tuples
                             produced by points_to_combos().
    resume : bool
        If True and the CSV already exists, skip combos that are already
        present (identified by the six provenance columns).
        If False (default), delete any existing CSV and start fresh.
    """
    # ── Build combo list ──────────────────────────────────────────────────────
    if isinstance(grid_or_combos, list):
        combos = grid_or_combos          # already flat tuples
    else:
        grid = grid_or_combos
        combos = list(itertools.product(
            grid["nmda_scale"],
            grid["gaba_scale"],
            grid["cue_rate"],
            grid["Jp"],
            grid["distractor_rates"],
            grid["seeds"],
        ))
    total = len(combos)

    # ── Optional resume: load already-completed combos ────────────────────────
    done_keys = set()
    if resume and os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        key_cols = ["nmda_scale", "gaba_scale", "cue_rate",
                    "Jp", "distractor_rate", "seed"]
        for _, r in existing.iterrows():
            done_keys.add(tuple(round(r[c], 6) if isinstance(r[c], float)
                                else r[c] for c in key_cols))
        print(f"[resume] {len(done_keys)} combos already done, skipping them.")
    elif os.path.exists(csv_path):
        os.remove(csv_path)

    t_start = time.time()
    n_done  = 0

    for (ns, gs, cr, jp, dr, seed) in combos:
        key = (round(ns, 6), round(gs, 6), cr, round(jp, 6), dr, seed)
        if key in done_keys:
            continue

        n_done += 1
        elapsed = time.time() - t_start
        eta_str = ""
        if n_done > 1:
            rate    = elapsed / (n_done - 1)
            remaining = rate * (total - len(done_keys) - n_done + 1)
            eta_str = f"  ETA≈{remaining/60:.1f} min"

        print(
            f"\r[{n_done}/{total - len(done_keys)}] "
            f"nmda={ns:.2f}  gaba={gs:.2f}  cue={cr} Hz  "
            f"Jp={jp:.2f}  distr={dr} Hz  seed={seed}"
            f"  ({elapsed:.0f}s){eta_str}",
            end="", flush=True
        )

        row = run_one(ns, gs, cr, jp, dr, seed, base_cfg)
        append_row_to_csv(row, csv_path)

    print()
    return pd.read_csv(csv_path)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_over_seeds(df, group_cols=None):
    """
    Average numeric metrics over seeds for each (nmda, gaba, cue, Jp,
    distractor_rate) combination.

    Returns a DataFrame with _mean and _std suffixed columns.
    """
    if group_cols is None:
        group_cols = ["nmda_scale", "gaba_scale",
                      "cue_rate", "Jp", "distractor_rate"]

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in group_cols + ["seed"]]

    agg = {}
    for c in numeric:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_std"]  = (c, "std")

    return df.groupby(group_cols).agg(**agg).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(df, metric="r1_delay", cue_r=200, jp=1.84,
                 distractor_rate=0, aggregate_seeds=True):
    """
    2-D heatmap of *metric* over (nmda_scale, gaba_scale).

    Parameters
    ----------
    distractor_rate : float – filter to this distractor rate (0 = no distractor)
    aggregate_seeds : bool  – if True, plot seed-averaged values
    """
    sub = df[
        (df["cue_rate"]         == cue_r) &
        (df["Jp"]               == jp)    &
        (df["distractor_rate"]  == distractor_rate)
    ].copy()

    if sub.empty:
        print(f"[heatmap] No data for cue={cue_r}, Jp={jp}, "
              f"distractor_rate={distractor_rate}")
        return

    if aggregate_seeds and "seed" in sub.columns:
        sub = sub.groupby(["nmda_scale", "gaba_scale"])[metric].mean().reset_index()

    try:
        pivot = sub.pivot(index="gaba_scale", columns="nmda_scale", values=metric)
    except Exception as e:
        print(f"[heatmap] pivot failed: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        extent=[
            pivot.columns.min() - 0.025,
            pivot.columns.max() + 0.025,
            pivot.index.min()   - 0.025,
            pivot.index.max()   + 0.025,
        ]
    )
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel("NMDA scale")
    ax.set_ylabel("GABA scale")
    ax.set_title(
        f"{metric}  |  cue={cue_r} Hz  Jp={jp}  "
        f"distr={'none' if distractor_rate == 0 else f'{distractor_rate} Hz'}"
    )
    plt.tight_layout()

    fname = (
        f"figures/heatmap_{metric}"
        f"_cue{cue_r}_jp{jp:.2f}"
        f"_distr{distractor_rate}.png"
    )
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved -> {fname}")


def plot_distractor_curve(df, nmda_s=1.00, gaba_s=1.00,
                          cue_r=200, jp=1.84,
                          metric="r1_delay"):
    """
    Plot *metric* as a function of distractor_rate for a fixed parameter point,
    with error bars across seeds.
    """
    sub = df[
        (np.isclose(df["nmda_scale"], nmda_s)) &
        (np.isclose(df["gaba_scale"], gaba_s)) &
        (df["cue_rate"] == cue_r)              &
        (np.isclose(df["Jp"], jp))
    ].copy()

    if sub.empty:
        print("[distractor_curve] No matching rows.")
        return

    grp = sub.groupby("distractor_rate")[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(grp["distractor_rate"], grp["mean"], yerr=grp["std"],
                marker="o", capsize=4)
    ax.set_xlabel("Distractor rate (Hz)  [0 = no distractor]")
    ax.set_ylabel(metric)
    ax.set_title(
        f"{metric} vs distractor strength\n"
        f"nmda={nmda_s}, gaba={gaba_s}, cue={cue_r} Hz, Jp={jp}"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fname = (
        f"figures/distr_curve_{metric}"
        f"_n{nmda_s:.2f}_g{gaba_s:.2f}_jp{jp:.2f}.png"
    )
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search over NMDA/GABA scales, cue rates, Jp, "
                    "distractor strengths, and random seeds."
    )
    p.add_argument("--quick",  action="store_true",
                   help="Use QUICK_GRID instead of FULL_GRID")
    p.add_argument("--points", action="store_true",
                   help="Use the explicit POINTS list instead of a grid")
    p.add_argument("--resume", action="store_true",
                   help="Skip combos already saved in the CSV")
    p.add_argument("--delay",  type=float, default=500,
                   help="Delay period (ms) [default: 500]")
    p.add_argument("--N",      type=int,   default=2000,
                   help="Total neuron count [default: 2000]")
    p.add_argument("--csv",    type=str,   default="results/grid_results.csv",
                   help="Output CSV path")
    # Allow overriding distractor_rates and seeds from CLI
    p.add_argument("--distractor_rates", type=float, nargs="+",
                   default=None,
                   help="Space-separated distractor rates in Hz "
                        "(0 = no distractor). Overrides grid default.")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=None,
                   help="Space-separated integer seeds. Overrides grid default.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Choose run mode ───────────────────────────────────────────────────────
    if args.points:
        # ── POINTS mode: explicit (nmda, gaba, Jp, cue) list ─────────────────
        pg = dict(POINTS_GRID)            # copy so we can override below

        if args.distractor_rates is not None:
            pg["distractor_rates"] = args.distractor_rates
        if args.seeds is not None:
            pg["seeds"] = args.seeds

        combos = points_to_combos(POINTS,
                                   pg["distractor_rates"],
                                   pg["seeds"])

        print(f"Points mode : {len(POINTS)} explicit points")
        print(f"  distractor_rates : {pg['distractor_rates']}")
        print(f"  seeds            : {pg['seeds']}")
        print(f"  Total simulations: {len(combos)}")
        print()
        for (ns, gs, cr, jp, _, _) in combos[:len(POINTS)]:   # show unique pts
            print(f"    nmda={ns}  gaba={gs}  Jp={jp}  cue={cr} Hz")
        print()

        run_source = combos          # pass the flat list directly

    else:
        # ── GRID mode: Cartesian product ──────────────────────────────────────
        grid = QUICK_GRID if args.quick else FULL_GRID

        if args.distractor_rates is not None:
            grid = dict(grid)
            grid["distractor_rates"] = args.distractor_rates
        if args.seeds is not None:
            grid = dict(grid)
            grid["seeds"] = args.seeds

        total_combos = (
            len(grid["nmda_scale"])       *
            len(grid["gaba_scale"])       *
            len(grid["cue_rate"])         *
            len(grid["Jp"])               *
            len(grid["distractor_rates"]) *
            len(grid["seeds"])
        )
        print(f"Grid mode  : {total_combos} simulations")
        print(f"  nmda_scale       : {grid['nmda_scale']}")
        print(f"  gaba_scale       : {grid['gaba_scale']}")
        print(f"  cue_rate         : {grid['cue_rate']}")
        print(f"  Jp               : {grid['Jp']}")
        print(f"  distractor_rates : {grid['distractor_rates']}")
        print(f"  seeds            : {grid['seeds']}")
        print()

        run_source = grid            # pass the dict; run_grid handles expansion

    # ── Base config ───────────────────────────────────────────────────────────
    base_cfg = dict(DEFAULT_CFG)
    base_cfg.update(
        baseline = 200 * ms,
        cue_dur  = 500 * ms,
        delay    = args.delay * ms,
        N_total  = args.N,
        plot     = False,
    )

    df = run_grid(run_source, base_cfg, csv_path=args.csv, resume=args.resume)

    print(f"\nSaved {len(df)} rows  →  {args.csv}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err:
            print(f"WARNING: {n_err} rows contain errors – inspect the CSV.")

    # ── Example plots (edit or remove as needed) ──────────────────────────────
    # Pick the first Jp value present in the data for heatmap
    jp_for_plot = float(df["Jp"].iloc[0]) if not df.empty else 1.84

    plot_heatmap(df, metric="r1_delay", cue_r=200, jp=jp_for_plot,
                 distractor_rate=0, aggregate_seeds=True)

    if 200 in df["distractor_rate"].values:
        plot_heatmap(df, metric="r1_delay", cue_r=200, jp=jp_for_plot,
                     distractor_rate=200, aggregate_seeds=True)

    # Distractor curve for the first unique (nmda, gaba) point
    if not df.empty:
        first = df.iloc[0]
        plot_distractor_curve(df,
                              nmda_s=first["nmda_scale"],
                              gaba_s=first["gaba_scale"],
                              cue_r=int(first["cue_rate"]),
                              jp=first["Jp"],
                              metric="r1_delay")

    # Seed-aggregated DataFrame
    agg_df = aggregate_over_seeds(df)
    agg_path = args.csv.replace(".csv", "_aggregated.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Aggregated (mean/std over seeds) → {agg_path}")


if __name__ == "__main__":
    main()
