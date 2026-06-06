# Working Memory Model — Schizophrenia & OCD Simulations

A biophysically realistic spiking neural network model of prefrontal-cortex working
memory, used to study how **NMDA conductance modulates attractor stability**, producing
schizophrenia-like instability (NMDA hypofunction) and OCD-like overstability
(NMDA hyperfunction).

Implemented in [Brian2](https://brian2.readthedocs.io/) following the integrate-and-fire
framework of **Brunel & Wang (2001)** as used by **Loh, Rolls & Deco (2007)**.

This repository accompanies the manuscript *"NMDA Modulates Working Memory Attractor
Stability: Opposite Regimes Can Produce Schizophrenia-like Instability and OCD-like
Overstability."* Full equations and parameters are in `supplement.md` / `supplement.pdf`.

---

## Overview

The model implements a cortical attractor network that:

- Encodes a cue into persistent delay-period activity (a memory attractor)
- Maintains memory via **NMDA-mediated recurrent excitation**, balanced by feedback inhibition
- Quantifies stability with **λ_delay**, the exponential decay rate of the selective
  differential firing rate Δr = r_S1 − r_S2 during the delay
- Simulates **schizophrenia** via NMDA hypofunction (shallow attractor → fast collapse)
  and **OCD** via NMDA hyperfunction (over-deepened attractor → pathological persistence)

### Network architecture

| Population | Size (N = 2000) | Role |
|---|---|---|
| Selective pool S1 | 240 (15% of E) | Encodes the cued memory |
| Selective pool S2 | 240 (15% of E) | Competing item / distractor target |
| Non-selective pool NS | 1120 (70% of E) | Background excitatory activity |
| Inhibitory pool I | 400 (20% of total) | Global feedback inhibition |

Synaptic receptors: **AMPA** (fast excitation) · **NMDA** (slow, voltage-gated, Mg²⁺-blocked) · **GABA_A** (inhibition).

---

## Conditions modelled

The three main-text regimes hold GABA fixed and vary NMDA:

| Condition key | NMDA scale | GABA scale | Interpretation |
|---|---|---|---|
| `control` | 1.00 | 1.00 | Healthy baseline |
| `-NMDA`   | 0.95 | 1.00 | **SCZ-like**: −5% NMDA (hypofunction) |
| `+NMDA`   | 1.10 | 1.00 | **OCD-like**: +10% NMDA (hyperfunction) |

Additional exploratory presets (used for the multi-seed grid, not the three-regime story)
are defined in `COND_PARAMS` inside `Working_Memory_Model.py`.

---

## Repository structure

```
.
├── Working_Memory_Model.py      # Core model: neurons, synapses, simulation, metrics
├── Run_Single_Trial.py          # One trial per condition → trace plots (Fig 1A–C)
├── Run_all_conditions.py        # Batch all conditions → CSV + comparison figures
├── Run_Grid_Trials.py           # Single-seed NMDA × GABA × Jp sweep (Fig 1D source)
├── Run_Grid_Trials_points.py    # Multi-seed sweep over selected points (Fig 2 source)
├── statistical_analysis.py      # Kruskal–Wallis / Mann–Whitney / Spearman + Fig 2 panels
├── lambda_delay_analysis.py     # λ_delay report (descriptives, KW, post-hoc, correlations)
├── seed_mean_sd_plot.py         # Mean ± SD across seeds plot
├── Data/
│   ├── Grid_results_150Hz.csv   # Single-seed grid, cue = 150 Hz
│   ├── Grid_results_200Hz.csv   # Single-seed grid, cue = 200 Hz  (Fig 1D)
│   └── Grid_results_250Hz.csv   # Single-seed grid, cue = 250 Hz
├── points_results_distractor.csv     # 19-seed points, distractor ON  (Fig 2 / main stats)
├── points_results_no_distractor.csv  # 19-seed points, distractor OFF
├── supplement.md                # Supplementary Information (MyST source)
├── supplement.pdf               # Supplementary Information (compiled)
├── FigureS1_NMDA_sweep.png      # Supplementary figure (NMDA sweep)
├── myst.yml                     # MyST build configuration
├── requirements.txt
├── LICENSE
├── figures/                     # Output figures (auto-created)
└── results/                     # Output CSVs (auto-created)
```

---

## Installation

```bash
git clone https://github.com/<your-username>/working-memory-model.git
cd working-memory-model

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> **Brian2** uses C++ code generation by default; if no compiler is available the model
> falls back to NumPy (`codegen_target="numpy"`, already the default in the configs).

---

## Usage

### Single trial (Fig 1A–C)

```bash
python Run_Single_Trial.py
```

Runs `control`, `-NMDA` and `+NMDA` at Jp = 1.84 and saves trace plots to `figures/`.
Edit the `conditions` list / `Jp` at the top of the file to change what is run.

### All conditions

```bash
python Run_all_conditions.py --no-plot
python Run_all_conditions.py --delay 200
python Run_all_conditions.py --condition=control --condition=-NMDA
```

> **Note:** condition names that begin with `-` must use the `--condition=-NMDA` form
> (an `=` sign), because `--condition -NMDA` is parsed as a missing argument.

Outputs go to `results/all_conditions.csv` and `figures/`.

### Single-seed grid sweep (Fig 1D)

```bash
python Run_Grid_Trials.py --quick          # small test grid
python Run_Grid_Trials.py --delay 500 --N 2000
```

Sweeps NMDA × GABA × Jp at a single seed and writes `results/grid_results.csv`
(the shipped `Data/Grid_results_*.csv` were produced this way, one file per cue rate).

### Multi-seed points + statistics (Fig 2)

```bash
# 1) generate the 19-seed points data (set distractor_rate and the seeds list inside the file)
python Run_Grid_Trials_points.py            # → results CSV (rename to points_results_*.csv)

# 2) run the analyses (set the input filename at the top of each script — see note below)
python statistical_analysis.py             # → statistical_report.txt, statistical_analysis.png
python lambda_delay_analysis.py            # → lambda_delay_report.txt, lambda_delay_analysis.png
python seed_mean_sd_plot.py                # → mean_sd_across_seeds.png
```

> **Before running the analysis scripts**, set their input path to the shipped data:
> - `statistical_analysis.py` → `DATA_PATH = 'points_results_distractor.csv'`
> - `seed_mean_sd_plot.py` → change the absolute path to a relative `'points_results_distractor.csv'`
> - `lambda_delay_analysis.py` → `pd.read_csv('points_results_distractor.csv')` (run again with
>   `points_results_no_distractor.csv` to regenerate the no-distractor report)
>
> The 19 seeds used for the published data were
> `[1, 7, 17, 19, 23, 29, 31, 37, 42, 43, 47, 54, 59, 65, 67, 93, 99, 123, 2024]`;
> set the `seeds` list in `Run_Grid_Trials_points.py` accordingly to reproduce exactly.

---

## Reproducing the figures

| Figure | Script | Data |
|---|---|---|
| Fig 1A–C (traces, Jp = 1.84) | `Run_Single_Trial.py` | live simulation |
| Fig 1D (λ_delay vs Jp, 3 regimes) | `Run_Grid_Trials.py` | `Data/Grid_results_200Hz.csv` |
| Fig 2A–C (multi-seed stats) | `statistical_analysis.py` | `points_results_distractor.csv` |
| Fig S1 (NMDA sweep) | `make_sup_fig` (see supplement) | `Data/Grid_results_200Hz.csv` |

---

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `N_total` | 2000 | Total neuron count |
| `Jp` | 1.84 | Intra-pool potentiation (swept 1.75–1.90) |
| `delay` | 500 ms | Delay period |
| `cue_rate` | 200 Hz | Cue Poisson rate to S1 |
| `nmda_scale` | 1.00 | NMDA conductance multiplier |
| `gaba_scale` | 1.00 | GABA conductance multiplier |
| `seed` | 42 | Random seed |

Full equations and the complete parameter set are in **`supplement.md` / `supplement.pdf`**.

---

## Output metrics

Each simulation returns (among others):

| Metric | Description |
|---|---|
| `r1_delay`, `r2_delay` | Mean S1 / S2 firing rate during the delay (Hz) |
| `persistent` | Whether a selective memory was maintained (rate-based) |
| `lambda_delay` | Exponential decay rate of Δr = r_S1 − r_S2 (NaN if no encoding / runaway) |
| `attractor_robustness` | Worst-case differential, min(Δr) |
| `frac_above_thr`, `selective_frac` | Fraction of the delay above threshold / selective |

---

## Notes on methods

- **Smoothing.** Population rates are smoothed with a **flat (rectangular) 50 ms window**
  (`smooth_rate(window="flat", width=50*ms)`) before fitting.
- **λ_delay fit window.** The exponential is fit over `[t_off + 100 ms, t_end − 50 ms]`,
  starting from the peak of Δr inside that window.
- **Single- vs multi-seed.** Fig 1D uses the single-seed grid (`Data/Grid_results_*.csv`);
  the multi-seed statistics (Fig 2) use 19 seeds at Jp = 1.75 and 1.88 over six NMDA/GABA
  points (`points_results_*.csv`).

---

## References

1. Brunel, N., & Wang, X.-J. (2001). Effects of neuromodulation in a cortical network model of object working memory dominated by recurrent inhibition. *J. Comput. Neurosci.* 11(1), 63–85.
2. Loh, M., Rolls, E. T., & Deco, G. (2007). A dynamical systems hypothesis of schizophrenia. *PLoS Comput. Biol.* 3(11), e228.
3. Wang, X.-J. (1999). Synaptic basis of cortical persistent activity: the importance of NMDA receptors to working memory. *J. Neurosci.* 19(21), 9587–9603.
4. Wang, X.-J. (2001). Synaptic reverberation underlying mnemonic persistent activity. *Trends Neurosci.* 24(8), 455–463.
5. Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. *Cereb. Cortex* 10(9), 910–923.
6. Stimberg, M., Brette, R., & Goodman, D. F. M. (2019). Brian 2, an intuitive and efficient neural simulator. *eLife* 8, e47314.

---

## Authors

- **Hamza Tahiri** — tahirihamza573@gmail.com
- **Fateme Amidizade** — fatemeamidizade@gmail.com
- **Razieh Zare** — zarerazieh98ir@gmail.com
- **Arvind Kumar** — arvind.k.panchal@gmail.com

Computational Neuroscience · Working Memory · Psychiatric Modelling — 2026

## License

MIT License — see `LICENSE`.
