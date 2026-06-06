"""
wm_model.py
===========
Biophysically realistic working memory model based on Wang (2001) /
Deco & Rolls (2007).

Architecture
------------
  - Leaky integrate-and-fire neurons
  - AMPA, NMDA (Mg²⁺-gated, voltage-dependent), GABA_A synaptic currents
  - Two selective excitatory pools (S1, S2) + non-selective pool (NS)
  - One global inhibitory interneuron pool
  - 800 background Poisson inputs per neuron

Conditions modelled
-------------------
  Control   : nmda_scale=1.00, gaba_scale=1.00
  SCZ-I     : –5 % NMDA          (shallow attractor, easy escape)
  SCZ-II    : –5 % NMDA, –10% GABA (attractor + disinhibition)
  SCZ-III   : –10% NMDA, –20% GABA (severe)
  OCD-I     : +8 % NMDA          (over-deep attractor, perseveration)
  OCD-II    : +20% NMDA          (extreme perseveration)
  Hypothetical treatment: +8% NMDA, +10% GABA

Author : Hamza
Date   : April 2026
"""

from brian2 import (
    NeuronGroup, Synapses, PoissonInput, PoissonGroup,
    SpikeMonitor, PopulationRateMonitor, StateMonitor,
    Network, network_operation, defaultclock, start_scope,
    ms, mV, nF, nS, Hz, kHz, mmole, pA, second, amp,
    prefs, run
)
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONDITION TABLE
# ============================================================
COND_PARAMS = {
    "control":       dict(nmda_scale=1.00, gaba_scale=1.00),
    # --- Schizophrenia ---
    "-NMDA":         dict(nmda_scale=0.95, gaba_scale=1.00),   # SCZ-I  : 5% NMDA reduction
    "--NMDA":        dict(nmda_scale=0.90, gaba_scale=1.00),   # SCZ-II : 10% NMDA reduction
    "-NMDA,-GABA":   dict(nmda_scale=0.95, gaba_scale=0.90),   # SCZ-III: 5% NMDA + 10% GABA
    "--NMDA,--GABA": dict(nmda_scale=0.90, gaba_scale=0.80),   # SCZ-IV : severe
    # --- OCD (over-stabilisation) ---
    "+NMDA":         dict(nmda_scale=1.1, gaba_scale=1.00),   # OCD-I  : 10% NMDA increase
    "++NMDA":        dict(nmda_scale=1.20, gaba_scale=1.00),   # OCD-II : 20% NMDA increase
    # --- GABA modulation alone ---
    "+GABA":         dict(nmda_scale=1.00, gaba_scale=1.10),
    "-GABA":         dict(nmda_scale=1.00, gaba_scale=0.90),
    "--GABA":        dict(nmda_scale=1.00, gaba_scale=0.80),
    # --- Hypothetical treatment ---
    "+NMDA,+GABA":   dict(nmda_scale=1.1, gaba_scale=1.10),
    "spt":   dict(nmda_scale=1.1, gaba_scale=0.2),
}


# ============================================================
# HELPERS
# ============================================================
def finalize_timing(cfg):
    """Fill in cue_on, cue_off, runtime from baseline + cue_dur + delay."""
    cfg = dict(cfg)
    if cfg["cue_on"] is None:
        cfg["cue_on"] = cfg["baseline"]
    if cfg["cue_off"] is None:
        cfg["cue_off"] = cfg["cue_on"] + cfg["cue_dur"]
    if cfg["runtime"] is None:
        cfg["runtime"] = cfg["baseline"] + cfg["cue_dur"] + cfg["delay"]
    return cfg


def mean_rate_in_window(R, t0, t1):
    """Mean smoothed population rate (Hz) in time window [t0, t1]."""
    t = np.asarray(R.t / second) * second
    r = np.asarray(R.smooth_rate(window="flat", width=50 * ms) / Hz)
    mask = (t >= t0) & (t < t1)
    return float(np.mean(r[mask])) if mask.sum() > 0 else 0.0


def compute_persistence_metrics(R1, R2, cue_off, runtime,
                                smooth_w=50 * ms, settle=100 * ms,
                                end_margin=50 * ms,
                                thr_hz=10.0, margin_hz=5.0):
    """
    Characterise delay-period activity.

    Returns dict with:
      r1_delay_mean, r1_delay_late, r1_delay_min,
      r2_delay_mean, r2_delay_late,
      frac_above_thr, selective_frac,
      collapse_time (s after delay onset, nan if no collapse),
      persistent (bool)
    """
    t  = np.asarray(R1.t / second)
    r1 = np.asarray(R1.smooth_rate(window="flat", width=smooth_w) / Hz)
    r2 = np.asarray(R2.smooth_rate(window="flat", width=smooth_w) / Hz)

    t0 = float((cue_off + settle) / second)
    t1 = float((runtime - end_margin) / second)
    mask = (t >= t0) & (t <= t1)

    if mask.sum() < 5:
        nan_dict = dict(r1_delay_mean=np.nan, r1_delay_late=np.nan,
                        r1_delay_min=np.nan, r2_delay_mean=np.nan,
                        r2_delay_late=np.nan, frac_above_thr=np.nan,
                        selective_frac=np.nan, collapse_time=np.nan,
                        persistent=False)
        return nan_dict

    td, y1, y2 = t[mask], r1[mask], r2[mask]

    late_mask        = td >= (td[-1] - 0.150)
    r1_delay_mean    = float(np.mean(y1))
    r2_delay_mean    = float(np.mean(y2))
    r1_delay_late    = float(np.mean(y1[late_mask]))
    r2_delay_late    = float(np.mean(y2[late_mask]))
    r1_delay_min     = float(np.min(y1))
    frac_above_thr   = float(np.mean(y1 > thr_hz))
    selective_frac   = float(np.mean(y1 > (y2 + margin_hz)))

    bad = (y1 <= thr_hz) | (y1 <= y2 + margin_hz)
    collapse_time    = float(td[np.argmax(bad)] - td[0]) if np.any(bad) else np.nan

    persistent = (
        (r1_delay_late > thr_hz) and
        (r1_delay_late > r2_delay_late + margin_hz) and
        (frac_above_thr > 0.8) and
        (selective_frac > 0.8)
    )

    # ── Extra richness ───────────────────────────────────────────────────────
    diff           = y1 - y2
    peak_diff      = float(np.max(diff))
    peak_diff_time = float(td[np.argmax(diff)] - td[0])   # s after delay onset
    r1_delay_std   = float(np.std(y1))
    r2_delay_std   = float(np.std(y2))

    # Early delay window: first 100 ms after settle
    early_mask   = (td - td[0]) <= 0.10
    r1_delay_early = float(np.mean(y1[early_mask])) if early_mask.sum() > 0 else np.nan

    # Attractor robustness: worst-case margin (minimum differential)
    attractor_robustness = float(np.min(diff))

    # Decay index: ratio late / early (< 1 = decaying, > 1 = growing)
    decay_index = (float(np.mean(y1[late_mask])) / (r1_delay_early + 1e-9))

    # Competitor suppression: how far below threshold is r2 kept?
    r2_delay_min   = float(np.min(y2))
    competitor_suppression = float(np.mean(thr_hz - y2))   # positive = well suppressed

    return dict(
        r1_delay_mean=r1_delay_mean, r1_delay_late=r1_delay_late,
        r1_delay_early=r1_delay_early, r1_delay_min=r1_delay_min,
        r1_delay_std=r1_delay_std,
        r2_delay_mean=r2_delay_mean, r2_delay_late=r2_delay_late,
        r2_delay_min=r2_delay_min, r2_delay_std=r2_delay_std,
        frac_above_thr=frac_above_thr, selective_frac=selective_frac,
        collapse_time=collapse_time, persistent=bool(persistent),
        peak_diff=peak_diff, peak_diff_time=peak_diff_time,
        attractor_robustness=attractor_robustness,
        decay_index=decay_index,
        competitor_suppression=competitor_suppression,
    )


def compute_lambda_delay(R_target, R_comp, t_start, t_end,
                         smooth_w=50 * ms,
                         min_peak_diff=3.0,
                         runaway_hz=200.0):
    """
    Fit an exponential decay to the (S1 - S2) differential during the delay.

    lambda > 0  -> differential decaying  -> SCZ-like instability
    lambda ~ 0  -> stable attractor       -> healthy control
    lambda < 0  -> differential growing   -> OCD-like hyper-stability
    nan         -> uncomputable: network never encoded, went epileptic,
                   or too few post-peak points. Treat as a separate failure
                   category in analysis -- NOT as lambda = 0.

    Fixes vs original
    -----------------
    Bug 1 (floor artifact)  : silent networks (diff always < 0) returned
                              lambda = 0 via np.maximum(..., 0.1) floor.
                              Now they return nan.
    Bug 2 (runaway guard)   : epileptic networks (gaba~0, rate > 200 Hz)
                              returned spurious lambda < 0. Now return nan.
    Bug 3 (peak-start bias) : fit now starts at argmax(diff), not at window
                              start, so the rising post-cue transient no
                              longer inflates lambda for persistent networks.
    """
    t  = np.asarray(R_target.t / second)
    rt = np.asarray(R_target.smooth_rate(window="flat", width=smooth_w) / Hz)
    rc = np.asarray(R_comp.smooth_rate(window="flat", width=smooth_w) / Hz)

    mask = (t >= float(t_start / second)) & (t <= float(t_end / second))
    if mask.sum() < 10:
        return np.nan, np.nan

    td   = t[mask]
    y1   = rt[mask]
    y2   = rc[mask]
    diff = y1 - y2
    x    = td - td[0]

    # Linear slope always computed (used by downstream metrics)
    linear_slope = float(np.polyfit(x, y1, 1)[0])

    # Guard 1: epileptic / runaway network
    if float(np.mean(y1)) > runaway_hz:
        return np.nan, linear_slope

    # Guard 2: network never encoded the cue
    if float(np.max(diff)) < min_peak_diff:
        return np.nan, linear_slope

    # Fix 3: start fit at peak of differential, not at window start.
    # Edge case: peak at end means differential is growing (OCD) -> use full window.
    peak_idx = int(np.argmax(diff))
    if peak_idx > int(0.80 * len(diff)):
        diff_post = diff
        x_post    = x
    else:
        diff_post = diff[peak_idx:]
        x_post    = x[peak_idx:]

    if len(diff_post) < 10:
        return np.nan, linear_slope

    y_fit        = np.maximum(diff_post, 0.01)
    p            = np.polyfit(x_post, np.log(y_fit), 1)
    lambda_delay = float(-p[0])
    return lambda_delay, linear_slope


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================
DEFAULT_CFG = dict(
    condition="control",

    # Timing
    baseline=200 * ms,
    cue_dur=500 * ms,
    delay=500 * ms,
    cue_on=None,
    cue_off=None,
    runtime=None,

    # Network size
    N_total=2000,
    f_inh=0.20,          # fraction inhibitory
    f_selective=0.15,    # fraction of E neurons in each selective pool

    # Single-neuron parameters
    El=-70 * mV,
    Vt=-50 * mV,
    Vr=-55 * mV,
    CmE=0.5 * nF,
    CmI=0.2 * nF,
    gLeakE=25 * nS,
    gLeakI=20 * nS,
    refE=2 * ms,
    refI=1 * ms,

    # Synaptic time constants
    tau_AMPA=2 * ms,
    tau_NMDA_rise=2 * ms,
    tau_NMDA_decay=100 * ms,
    tau_GABA=10 * ms,
    alpha_NMDA=0.5 * kHz,

    # Reversal potentials
    V_E=0 * mV,
    V_I=-70 * mV,
    Mg_conc=1.0 * mmole,

    # Background Poisson input
    N_ext=800,
    rate_ext_E=3 * Hz,
    rate_ext_I=3 * Hz,
    gextE=2.1 * nS,
    gextI=1.62 * nS,

    # Base synaptic conductances (calibrated for N_total=2000)
    gEEA_base=0.05 * nS,
    gEIA_base=0.04 * nS,
    gEEN_base=0.165 * nS,
    gEIN_base=0.13 * nS,
    gIE_base=1.3 * nS,
    gII_base=1.0 * nS,

    # Structured connectivity: intra-pool potentiation factor
    Jp=1.84,

    # Cue stimulus
    cue_rate=200 * Hz,
    cue_pool=0,
    cue_mode="one_to_one",
    cue_g=None,

    # Distractor list – each entry is a dict:
    #   dict(t_on=..., t_off=..., rate=..., pool=<0|1|2|'I'>, mode='one_to_one')
    distractors=[dict(t_on=800, t_off=1100, rate=200, pool=1, mode='one_to_one')],

    # Simulation
    dt=0.02 * ms,
    seed=42,
    codegen_target="numpy",
)


# ============================================================
# MAIN SIMULATION FUNCTION
# ============================================================
def run_wm_combined(cfg, overrides=None, plot=True, report="text"):
    """
    Build and run a single working-memory simulation trial.

    Parameters
    ----------
    cfg      : dict  – simulation configuration (see DEFAULT_CFG)
    overrides: dict  – optional key-value pairs that override cfg
    plot     : bool  – show raster / rate / NMDA / GABA figures
    report   : str   – Brian2 progress reporting ('text' | None)

    Returns
    -------
    cfg_used  : dict            – full resolved configuration
    monitors  : dict            – all Brian2 monitor objects
    metrics   : dict            – quantitative analysis results
    """
    cfg_used = dict(cfg)
    if overrides:
        cfg_used.update(overrides)
    cfg_used = finalize_timing(cfg_used)

    start_scope()
    np.random.seed(cfg_used["seed"])
    defaultclock.dt = cfg_used["dt"]
    prefs.codegen.target = cfg_used["codegen_target"]

    # ── Derived sizes ─────────────────────────────────────────────────────────
    NE   = int(cfg_used["N_total"] * (1.0 - cfg_used["f_inh"]))
    NI   = int(cfg_used["N_total"] * cfg_used["f_inh"])
    subN = int(cfg_used["f_selective"] * NE)

    Jp   = cfg_used["Jp"]
    fsel = cfg_used["f_selective"]
    Jm   = 1.0 - fsel * (Jp - 1.0) / (1.0 - fsel)   # keeps mean input constant

    cond = cfg_used["condition"]
    if cond not in COND_PARAMS:
        raise ValueError(f"Unknown condition '{cond}'. "
                         f"Valid options: {list(COND_PARAMS.keys())}")

    nmda_scale = COND_PARAMS[cond]["nmda_scale"]
    gaba_scale = COND_PARAMS[cond]["gaba_scale"]

    # Scale conductances for actual network size
    scale_E = 1600.0 / NE
    scale_I = 400.0  / NI

    gEEA = cfg_used["gEEA_base"] * scale_E
    gEIA = cfg_used["gEIA_base"] * scale_E
    gEEN = cfg_used["gEEN_base"] * scale_E * nmda_scale
    gEIN = cfg_used["gEIN_base"] * scale_E * nmda_scale
    gIE  = cfg_used["gIE_base"]  * scale_I * gaba_scale
    gII  = cfg_used["gII_base"]  * scale_I * gaba_scale

    print(f"[COND] {cond}  →  nmda_scale={nmda_scale:.3f}, gaba_scale={gaba_scale:.3f}")
    print(f"[EFF]  gEEA={gEEA/nS:.4f} nS  gEEN={gEEN/nS:.4f} nS  gIE={gIE/nS:.4f} nS")

    # ── Neuron equations ──────────────────────────────────────────────────────
    El  = cfg_used["El"];  Vt = cfg_used["Vt"];  Vr = cfg_used["Vr"]
    CmE = cfg_used["CmE"]; CmI = cfg_used["CmI"]
    gLeakE = cfg_used["gLeakE"]; gLeakI = cfg_used["gLeakI"]
    V_E = cfg_used["V_E"]; V_I = cfg_used["V_I"]
    tau_AMPA = cfg_used["tau_AMPA"]
    tau_NMDA_rise  = cfg_used["tau_NMDA_rise"]
    tau_NMDA_decay = cfg_used["tau_NMDA_decay"]
    tau_GABA = cfg_used["tau_GABA"]
    alpha    = cfg_used["alpha_NMDA"]
    Mg_conc  = cfg_used["Mg_conc"]

    eqsE = """
    label : integer (constant)

    dV/dt = (-gLeakE*(V - El) - I_AMPA - I_NMDA - I_GABA - I_AMPA_ext + I_input) / CmE : volt (unless refractory)

    I_AMPA      = s_AMPA     * (V - V_E)                                            : amp
    ds_AMPA/dt  = -s_AMPA    / tau_AMPA                                              : siemens

    I_GABA      = s_GABA     * (V - V_I)                                            : amp
    ds_GABA/dt  = -s_GABA    / tau_GABA                                              : siemens

    I_AMPA_ext     = s_AMPA_ext * (V - V_E)                                         : amp
    ds_AMPA_ext/dt = -s_AMPA_ext / tau_AMPA                                         : siemens

    ds_NMDA/dt  = -s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA)              : 1
    dx/dt       = -x / tau_NMDA_rise                                                 : 1

    I_NMDA      = gEEN * s_NMDA_tot * (V - V_E) / (1 + exp(-0.062*V/mvolt) * (Mg_conc/mmole / 3.57)) : amp
    s_NMDA_tot  : 1

    I_input : amp
    """

    eqsI = """
    dV/dt = (-gLeakI*(V - El) - I_AMPA - I_NMDA - I_GABA - I_AMPA_ext) / CmI : volt (unless refractory)

    I_AMPA      = s_AMPA     * (V - V_E)                                            : amp
    ds_AMPA/dt  = -s_AMPA    / tau_AMPA                                              : siemens

    I_GABA      = s_GABA     * (V - V_I)                                            : amp
    ds_GABA/dt  = -s_GABA    / tau_GABA                                              : siemens

    I_AMPA_ext     = s_AMPA_ext * (V - V_E)                                         : amp
    ds_AMPA_ext/dt = -s_AMPA_ext / tau_AMPA                                         : siemens

    I_NMDA      = gEIN * s_NMDA_tot * (V - V_E) / (1 + exp(-0.062*V/mvolt) * (Mg_conc/mmole / 3.57)) : amp
    s_NMDA_tot  : 1
    """

    # ── Populations ───────────────────────────────────────────────────────────
    popE = NeuronGroup(NE, eqsE, threshold="V > Vt",
                       reset="V = Vr; x += 1",
                       refractory=cfg_used["refE"], method="euler")
    popI = NeuronGroup(NI, eqsI, threshold="V > Vt",
                       reset="V = Vr",
                       refractory=cfg_used["refI"], method="euler")

    popE1 = popE[:subN]           # selective pool S1 (cued)
    popE2 = popE[subN:2 * subN]   # selective pool S2 (competitor)
    popE3 = popE[2 * subN:]       # non-selective pool

    popE1.label = 0;  popE2.label = 1;  popE3.label = 2
    popE.V = Vr + 2 * mV * np.random.rand(NE)
    popI.V = Vr + 2 * mV * np.random.rand(NI)
    popE.s_NMDA_tot = tau_NMDA_decay * 5 * Hz * 0.2
    popI.s_NMDA_tot = tau_NMDA_decay * 5 * Hz * 0.2
    popE.I_input = 0 * amp

    # ── Recurrent AMPA ────────────────────────────────────────────────────────
    C_EE_AMPA = Synapses(popE, popE, "w : siemens",
                         on_pre="s_AMPA += w", delay=0.5 * ms)
    C_EE_AMPA.connect(condition="i != j")
    C_EE_AMPA.w = gEEA
    C_EE_AMPA.w["label_pre == label_post and label_pre < 2"] = gEEA * Jp
    C_EE_AMPA.w["label_pre != label_post and label_post < 2"] = gEEA * Jm

    C_EI_AMPA = Synapses(popE, popI, "w : siemens",
                         on_pre="s_AMPA += w", delay=0.5 * ms)
    C_EI_AMPA.connect();  C_EI_AMPA.w = gEIA

    C_IE = Synapses(popI, popE, "w : siemens",
                    on_pre="s_GABA += w", delay=0.5 * ms)
    C_IE.connect();  C_IE.w = gIE

    C_II = Synapses(popI, popI, "w : siemens",
                    on_pre="s_GABA += w", delay=0.5 * ms)
    C_II.connect(condition="i != j");  C_II.w = gII

    # ── Pooled NMDA (drives actual dynamics) ──────────────────────────────────
    NMDA_sum_group = NeuronGroup(3, "s : 1")

    NMDA_sum = Synapses(popE, NMDA_sum_group,
                        "s_post = s_NMDA_pre : 1 (summed)")
    NMDA_sum.connect(j="label_pre")

    NMDA_set_total_E = Synapses(
        NMDA_sum_group, popE,
        """
        w : 1 (constant)
        s_NMDA_tot_post = w * s_pre : 1 (summed)
        """
    )
    NMDA_set_total_E.connect()
    NMDA_set_total_E.w = 1.0
    NMDA_set_total_E.w["i == label_post and label_post < 2"] = Jp
    NMDA_set_total_E.w["i != label_post and label_post < 2"] = Jm

    NMDA_set_total_I = Synapses(NMDA_sum_group, popI,
                                "s_NMDA_tot_post = s_pre : 1 (summed)")
    NMDA_set_total_I.connect()

    # ── Diagnostic NMDA map (no current injection) ────────────────────────────
    C_EE_NMDA_map = Synapses(popE, popE, "w : 1")
    C_EE_NMDA_map.connect(condition="i != j")
    C_EE_NMDA_map.w = 1.0
    C_EE_NMDA_map.w["label_pre == label_post and label_pre < 2"] = Jp
    C_EE_NMDA_map.w["label_pre != label_post and label_post < 2"] = Jm

    C_EI_NMDA_map = Synapses(popE, popI, "w : 1")
    C_EI_NMDA_map.connect();  C_EI_NMDA_map.w = 1.0

    gaba_map = dict(
        IE_i=np.array(C_IE.i[:]), IE_j=np.array(C_IE.j[:]),
        IE_w=np.full(len(C_IE.i[:]), float(gIE / nS)),
        II_i=np.array(C_II.i[:]), II_j=np.array(C_II.j[:]),
        II_w=np.full(len(C_II.i[:]), float(gII / nS)),
    )

    # ── Background Poisson input ──────────────────────────────────────────────
    extinputE = PoissonInput(popE, "s_AMPA_ext", cfg_used["N_ext"],
                             cfg_used["rate_ext_E"], cfg_used["gextE"])
    extinputI = PoissonInput(popI, "s_AMPA_ext", cfg_used["N_ext"],
                             cfg_used["rate_ext_I"], cfg_used["gextI"])

    # ── Cue stimulus ──────────────────────────────────────────────────────────
    cue_g      = cfg_used["gextE"] if cfg_used["cue_g"] is None else cfg_used["cue_g"]
    cue_group  = PoissonGroup(subN, rates=0 * Hz)
    pool_map   = {0: popE1, 1: popE2, 2: popE3}
    cue_target = pool_map[int(cfg_used["cue_pool"])]

    C_cue = Synapses(cue_group, cue_target, "w : siemens",
                     on_pre="s_AMPA_ext += w")
    if cfg_used["cue_mode"] == "one_to_one":
        C_cue.connect(j="i")
    else:
        C_cue.connect()
    C_cue.w = cue_g

    cue_on  = cfg_used["cue_on"]
    cue_off = cfg_used["cue_off"]

    # ── Distractors ───────────────────────────────────────────────────────────
    distractors   = cfg_used.get("distractors", [])
    distr_groups  = []

    for ev in distractors:
        if ev.get("pool") == "I":
            tgt   = popI;   Nsrc = NI;   default_g = cfg_used["gextI"]
        else:
            pool_id = int(ev["pool"])
            tgt     = pool_map[pool_id]
            Nsrc    = len(tgt)
            default_g = cfg_used["gextE"]

        g_ev = ev.get("g", default_g)
        pg   = PoissonGroup(Nsrc, rates=0 * Hz)
        syn  = Synapses(pg, tgt, "w : siemens", on_pre="s_AMPA_ext += w")
        if ev.get("mode", "one_to_one") == "one_to_one" and len(tgt) == Nsrc:
            syn.connect(j="i")
        else:
            syn.connect()
        syn.w = g_ev
        distr_groups.append((pg, ev))

    @network_operation(dt=1 * ms)
    def event_controller():
        cue_group.rates = cfg_used["cue_rate"] if (cue_on <= defaultclock.t < cue_off) else 0 * Hz
        for pg, ev in distr_groups:
            # normalise t_on/t_off/rate: accept bare numbers (ms/Hz) or Brian2 quantities
            t_on  = ev["t_on"]  if hasattr(ev["t_on"],  "dimensions") else ev["t_on"]  * ms
            t_off = ev["t_off"] if hasattr(ev["t_off"], "dimensions") else ev["t_off"] * ms
            rate  = ev["rate"]  if hasattr(ev["rate"],  "dimensions") else ev["rate"]  * Hz
            pg.rates = rate if (t_on <= defaultclock.t < t_off) else 0 * Hz

    # ── State monitors (diagnostics) ──────────────────────────────────────────
    E_diag_idx = [i for i in [0, subN, 2 * subN] if i < NE]
    I_diag_idx = [0, min(5, NI - 1)]
    E_state = StateMonitor(popE, ["I_GABA", "I_NMDA", "I_AMPA", "I_AMPA_ext", "V"],
                           record=E_diag_idx, dt=1 * ms)
    I_state = StateMonitor(popI, ["I_GABA", "I_NMDA", "I_AMPA", "I_AMPA_ext", "V"],
                           record=I_diag_idx, dt=1 * ms)

    nmda_rec_idx  = [i for i in [0, 5, 10, 35] if i < len(popE1)] or [0]
    NMDA_trace    = StateMonitor(popE1, ["s_NMDA", "s_NMDA_tot", "x", "V"],
                                 record=nmda_rec_idx, dt=1 * ms)

    # ── Spike & rate monitors ─────────────────────────────────────────────────
    SME1 = SpikeMonitor(popE1, record=True)
    SME2 = SpikeMonitor(popE2, record=True)
    SME3 = SpikeMonitor(popE3, record=True)
    SMI  = SpikeMonitor(popI,  record=True)
    R1   = PopulationRateMonitor(popE1)
    R2   = PopulationRateMonitor(popE2)
    R3   = PopulationRateMonitor(popE3)
    RI   = PopulationRateMonitor(popI)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"Network : NE={NE}, NI={NI}, subN={subN}  |  Jp={Jp:.2f}, Jm={Jm:.3f}")
    print(f"Timing  : baseline={cfg_used['baseline']/ms:.0f} ms  "
          f"cue=[{cue_on/ms:.0f}–{cue_off/ms:.0f}] ms  "
          f"runtime={cfg_used['runtime']/ms:.0f} ms")
    for ev in distractors:
        _ton  = float(ev['t_on']  / ms if hasattr(ev['t_on'],  'dimensions') else ev['t_on'])
        _toff = float(ev['t_off'] / ms if hasattr(ev['t_off'], 'dimensions') else ev['t_off'])
        _rate = float(ev['rate']  / Hz if hasattr(ev['rate'],  'dimensions') else ev['rate'])
        print(f"  distractor pool={ev.get('pool')}  "
              f"[{_ton:.0f}–{_toff:.0f}] ms  "
              f"rate={_rate:.1f} Hz")
    print("=" * 72)

    run(cfg_used["runtime"], report=report)

    # ── Analysis ──────────────────────────────────────────────────────────────
    baseline_win = (100 * ms, cfg_used["baseline"])
    delay_win    = (cue_off + 100 * ms, cfg_used["runtime"] - 100 * ms)

    r1_base  = mean_rate_in_window(R1, *baseline_win)
    r1_cue   = mean_rate_in_window(R1, cue_on + 50 * ms, cue_off - 50 * ms)
    r1_delay = mean_rate_in_window(R1, *delay_win)
    r2_delay = mean_rate_in_window(R2, *delay_win)
    r3_delay = mean_rate_in_window(R3, *delay_win)
    ri_delay = mean_rate_in_window(RI, *delay_win)

    persist_metrics  = compute_persistence_metrics(R1, R2, cue_off,
                                                   cfg_used["runtime"])
    lambda_d, lin_d  = compute_lambda_delay(R1, R2,
                                            cue_off + 100 * ms,
                                            cfg_used["runtime"] - 50 * ms)

    # ── Cue-period peak ───────────────────────────────────────────────────────
    t_arr  = np.asarray(R1.t / second)
    r1_arr = np.asarray(R1.smooth_rate(window="flat", width=50 * ms) / Hz)
    r2_arr = np.asarray(R2.smooth_rate(window="flat", width=50 * ms) / Hz)
    r3_arr = np.asarray(R3.smooth_rate(window="flat", width=50 * ms) / Hz)
    ri_arr = np.asarray(RI.smooth_rate(window="flat", width=50 * ms) / Hz)

    cue_mask = (t_arr >= float((cue_on + 50 * ms) / second)) &                (t_arr <= float((cue_off - 50 * ms) / second))
    r1_cue_peak = float(np.max(r1_arr[cue_mask])) if cue_mask.sum() > 0 else np.nan

    # Distractor window metrics
    distr_list = cfg_used.get("distractors", [])
    if distr_list:
        ev = distr_list[0]
        d_on  = ev["t_on"]  if hasattr(ev["t_on"],  "dimensions") else ev["t_on"]  * ms
        d_off = ev["t_off"] if hasattr(ev["t_off"], "dimensions") else ev["t_off"] * ms
        d_mask = (t_arr >= float(d_on / second)) & (t_arr <= float(d_off / second))
        pre_mask = (t_arr >= float((d_on - 150 * ms) / second)) &                    (t_arr <  float(d_on / second))
        post_mask = (t_arr >= float(d_off / second)) &                     (t_arr <= float((d_off + 150 * ms) / second))
        r1_during_distr  = float(np.mean(r1_arr[d_mask]))   if d_mask.sum()   > 0 else np.nan
        r1_pre_distr     = float(np.mean(r1_arr[pre_mask]))  if pre_mask.sum() > 0 else np.nan
        r1_post_distr    = float(np.mean(r1_arr[post_mask])) if post_mask.sum()> 0 else np.nan
        distractor_drop  = r1_pre_distr - r1_during_distr    # positive = S1 dropped
        distractor_recovery = r1_post_distr - r1_during_distr  # positive = recovered
    else:
        r1_during_distr = r1_pre_distr = r1_post_distr = np.nan
        distractor_drop = distractor_recovery = np.nan

    # SNR: selectivity ratio during delay
    snr_delay = r1_delay / (r2_delay + 1e-9)

    # Cue gain and delay gain relative to baseline
    cue_gain   = r1_cue   / (r1_base + 1e-9)
    delay_gain = r1_delay / (r1_base + 1e-9)

    # lambda validity flag
    lambda_valid = not (lambda_d != lambda_d)   # True if not nan

    # Network regime label
    if r1_delay > 200:
        regime = "runaway"
    elif r1_delay > r2_delay + 3.0 and r1_delay > 10.0:
        regime = "persistent"
    elif r1_delay > 5.0:
        regime = "transient"
    else:
        regime = "silent"

    metrics = dict(
        # ── Baseline / cue ───────────────────────────────────────────────────
        r1_base=r1_base, r1_cue=r1_cue, r1_cue_peak=r1_cue_peak,
        cue_gain=cue_gain,
        cue_response=(r1_cue > 2.0 * r1_base) if r1_base > 0 else (r1_cue > 5.0),
        # ── Delay period population rates ────────────────────────────────────
        r1_delay=r1_delay, r2_delay=r2_delay,
        r3_delay=r3_delay, ri_delay=ri_delay,
        delay_gain=delay_gain,
        snr_delay=snr_delay,
        delay_minus_competitor=r1_delay - r2_delay,
        delay_minus_ns=r1_delay - r3_delay,
        # ── Stability / attractor metrics ────────────────────────────────────
        lambda_delay=lambda_d, lambda_valid=lambda_valid,
        linear_delay_slope=lin_d,
        inhibitory_balance=ri_delay / (r1_delay + 1e-9),
        # ── Distractor metrics ───────────────────────────────────────────────
        r1_during_distr=r1_during_distr,
        r1_pre_distr=r1_pre_distr,
        r1_post_distr=r1_post_distr,
        distractor_drop=distractor_drop,
        distractor_recovery=distractor_recovery,
        survives_distractor=float(r1_delay > r2_delay + 3.0 and r1_delay > 5.0),
        # ── Regime label ─────────────────────────────────────────────────────
        regime=regime,
    )
    metrics.update(persist_metrics)

    print("\nFIRING RATES (Hz)")
    print(f"  S1  base={r1_base:.2f}  cue={r1_cue:.2f}  delay={r1_delay:.2f}")
    print(f"  S2  delay={r2_delay:.2f}  |  NS delay={r3_delay:.2f}  |  I delay={ri_delay:.2f}")
    print(f"  persistence={metrics['persistent']}  "
          f"λ_delay={lambda_d:.3f}  "
          f"frac_above_thr={metrics['frac_above_thr']:.2f}")

    monitors = dict(
        SME1=SME1, SME2=SME2, SME3=SME3, SMI=SMI,
        R1=R1, R2=R2, R3=R3, RI=RI,
        NMDA_trace=NMDA_trace,
        E_state=E_state, I_state=I_state,
        nmda_map=dict(
            EE_nmda_i=np.array(C_EE_NMDA_map.i[:]),
            EE_nmda_j=np.array(C_EE_NMDA_map.j[:]),
            EE_nmda_w=np.array(C_EE_NMDA_map.w[:]),
        ),
        gaba_map=gaba_map,
        C_EE_NMDA_map=C_EE_NMDA_map,
        C_IE=C_IE, C_II=C_II,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plot:
        _plot_trial(cfg_used, monitors, metrics,
                    cue_on, cue_off, subN, nmda_rec_idx, gIE, gII, NE, NI)

    return cfg_used, monitors, metrics


def _plot_trial(cfg_used, monitors, metrics,
                cue_on, cue_off, subN, nmda_rec_idx, gIE, gII, NE, NI):

    SME1 = monitors["SME1"];  SME2 = monitors["SME2"]
    SME3 = monitors["SME3"];  R1 = monitors["R1"]
    R2 = monitors["R2"];      R3 = monitors["R3"]
    RI = monitors["RI"];      NMDA_trace = monitors["NMDA_trace"]

    cond = cfg_used["condition"]
    Jp   = cfg_used["Jp"]

    # 🔥 Balanced figure (KEY FIX)
    fig, axs = plt.subplots(
        5, 1,
        figsize=(20, 14),   # ↓ less tall, more readable
        sharex=True
    )

    plt.subplots_adjust(
        hspace=0.35,        # space between plots
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.92
    )

    span_kw = dict(alpha=0.15, color="gold")

    title_fs = 11
    label_fs = 10
    tick_fs  = 9

    # =========================
    # 1. Raster plots
    # =========================
    for ax, sm, lbl in zip(axs[:3], [SME1, SME2, SME3], ["S1", "S2", "NS"]):
        ax.plot(sm.t / ms, sm.i, ".", markersize=0.5, alpha=0.5)
        ax.axvspan(cue_on / ms, cue_off / ms, **span_kw)
        ax.set_title(f"{lbl} raster", fontsize=title_fs)
        ax.set_ylabel("Neuron #", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.margins(x=0)

    # =========================
    # 2. Population rates
    # =========================
    for R, lbl, ls in [(R1, "S1", "-"), (R2, "S2", "-"),
                       (R3, "NS", "-"), (RI, "I", "--")]:
        axs[3].plot(
            R.t / ms,
            R.smooth_rate(window="flat", width=50 * ms) / Hz,
            linestyle=ls,
            label=lbl
        )

    axs[3].axvspan(cue_on / ms, cue_off / ms, **span_kw)

    axs[3].set_title(
        f"Population rates | persistent={metrics['persistent']} | λ={metrics['lambda_delay']:.3f}",
        fontsize=title_fs
    )

    axs[3].set_ylabel("Hz", fontsize=label_fs)
    axs[3].grid(alpha=0.3)
    axs[3].tick_params(labelsize=tick_fs)

    # 🔥 legend inside but clean
    axs[3].legend(
        loc="upper right",
        fontsize=9,
        frameon=True
    )

    # =========================
    # 3. NMDA traces
    # =========================
    for k, rec_i in enumerate(nmda_rec_idx):
        axs[4].plot(
            NMDA_trace.t / ms,
            NMDA_trace.s_NMDA[k],
            label=f"S1[{rec_i}]"
        )

    axs[4].plot(
        NMDA_trace.t / ms,
        NMDA_trace.s_NMDA_tot[0],
        lw=2,
        label="NMDA total"
    )

    axs[4].axvspan(cue_on / ms, cue_off / ms, **span_kw)

    axs[4].set_title("NMDA gating", fontsize=title_fs)
    axs[4].set_ylabel("NMDA", fontsize=label_fs)
    axs[4].set_xlabel("Time (ms)", fontsize=label_fs)

    axs[4].grid(alpha=0.3)
    axs[4].tick_params(labelsize=tick_fs)

    axs[4].legend(
        loc="upper right",
        fontsize=9,
        frameon=True
    )

    # =========================
    # Global title
    # =========================
    fig.suptitle(
        f"WM | {cond} | Jp={Jp:.2f} | cue={cfg_used['cue_rate']/Hz:.0f} Hz",
        fontsize=13
    )

    # =========================
    # Save (clean)
    # =========================
    out = f"figures/wm_{cond.replace(',', '_')}.png"
    plt.savefig(out, dpi=200)
    print(f"Saved -> {out}")

    plt.show()