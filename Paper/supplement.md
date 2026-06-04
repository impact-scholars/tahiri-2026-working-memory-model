---
title: 'Supplementary Information — NMDA Modulates Working Memory Attractor Stability'
short_title: Supplementary Information
subtitle: 'Opposite Regimes Can Produce Schizophrenia-like Instability and OCD-like Overstability'
authors:
  - name: H. Tahiri
  - name: R. Zare
  - name: F. Amidizade
  - name: I. Chakir
  - name: A. Kumar
exports:
  - format: typst
numbering:
  headings: true
  equation: true
  figure: true
  table: true
bibliography: bib.bib
abstract: |
  This document contains the supplementary figures, tables, and extended methods supporting the main manuscript regarding NMDA modulation of working memory attractors.
---

# Supplementary Figures
This supplement gives the complete mathematical description of the spiking network ([](#sec-model)), the exact parameter values used in all simulations ([](#tab-neuron)–[](#tab-arch)), the definition of the $\lambda_{\mathrm{delay}}$ metric ([](#sec-lambda)), the multi-seed statistical protocol ([](#sec-stats)), a supplementary NMDA-conductance sweep ([](#fig-nmda-sweep)), and the full statistical tables ([](#sec-tables)). The model follows the integrate-and-fire framework of  {cite:p}`loh2007dynamical` as used by 
{cite:p}`loh2007dynamical`,and was implemented in Brian2 {cite:p}`stimberg2019brian2`.

(sec-model)=
## S1. Network model

### S1.1 Single-neuron dynamics

Both excitatory ($E$) and inhibitory ($I$) cells are leaky integrate-and-fire neurons. The subthreshold membrane potential $V$ obeys

```{math}
:label: eq-membrane
C_m\,\frac{dV}{dt} = -g_L\,(V-E_L) - I_{\mathrm{syn}}(t),
```

with cell-type-specific $C_m$ and $g_L$ ([](#tab-neuron)). When $V$ reaches the threshold $V_{\mathrm{th}}$ a spike is emitted and $V$ is clamped to the reset $V_{r}$ for an absolute refractory period $t_{\mathrm{ref}}$. The total synaptic current is

```{math}
:label: eq-isyn
I_{\mathrm{syn}}(t) = I_{\mathrm{AMPA,ext}}(t) + I_{\mathrm{AMPA,rec}}(t) + I_{\mathrm{NMDA,rec}}(t) + I_{\mathrm{GABA}}(t).
```

### S1.2 Synaptic currents

AMPA (external and recurrent) and GABA$_{\mathrm A}$ currents are

```{math}
:label: eq-ampa-gaba
\begin{aligned}
I_{\mathrm{AMPA}}(t) &= s_{\mathrm{AMPA}}(t)\,\big(V(t)-V_E\big), &
\frac{ds_{\mathrm{AMPA}}}{dt} &= -\frac{s_{\mathrm{AMPA}}}{\tau_{\mathrm{AMPA}}} + \sum_k g_{\mathrm{AMPA}}\,\delta(t-t_k),\\[2pt]
I_{\mathrm{GABA}}(t) &= s_{\mathrm{GABA}}(t)\,\big(V(t)-V_I\big), &
\frac{ds_{\mathrm{GABA}}}{dt} &= -\frac{s_{\mathrm{GABA}}}{\tau_{\mathrm{GABA}}} + \sum_k g_{\mathrm{GABA}}\,\delta(t-t_k),
\end{aligned}
```

where each presynaptic spike at $t_k$ increments the conductance variable by the corresponding synaptic weight. The NMDA current carries a voltage-dependent Mg$^{2+}$ block {cite:p}`jahr1990voltage`:

```{math}
:label: eq-nmda
I_{\mathrm{NMDA}}(t) = \frac{g_{\mathrm{NMDA}} s^{\mathrm{tot}}_{\mathrm{NMDA}}(t) (V(t)-V_E)}{1 + \frac{[\mathrm{Mg}^{2+}]}{3.57} \exp(-0.062 V(t))},
```
with $V$ in mV and $[\mathrm{Mg}^{2+}]$ in mM. The NMDA gating variable has a two-stage (rise/decay) form,

```{math}
:label: eq-nmda-gating
\begin{aligned}
\frac{ds_{\mathrm{NMDA}}}{dt} &= -\frac{s_{\mathrm{NMDA}}}{\tau_{\mathrm{NMDA,decay}}} + \alpha x(t) (1-s_{\mathrm{NMDA}}), \\[6pt]
\frac{dx}{dt} &= -\frac{x}{\tau_{\mathrm{NMDA,rise}}} + \sum_k \delta(t-t_k),
\end{aligned}
```

and $s^{\mathrm{tot}}_{\mathrm{NMDA}}$ denotes the gating summed over the presynaptic excitatory pool. All synapses act with a transmission delay of $0.5\ \mathrm{ms}$.



### S1.3 Architecture and connectivity

The network contains $N=2000$ neurons: $N_E=1600$ excitatory ($f_{\mathrm{inh}}=0.20$) and $N_I=400$ inhibitory. The excitatory population is split into two selective pools $S_1$ and $S_2$ of $240$ neurons each ($f_{\mathrm{sel}}=0.15$ of $N_E$) and a non-selective pool (NS) of $1120$ neurons. The inhibitory pool provides global feedback inhibition. Recurrent excitation is structured by a within-pool potentiation factor $J_p$ and a compensating cross-pool factor

```{math}
:label: eq-jm
J_m = 1 - f_{\mathrm{sel}}\,\frac{J_p-1}{1-f_{\mathrm{sel}}},
```

which keeps the mean recurrent input constant as $J_p$ varies. Excitatory–excitatory AMPA weights are $g_{\mathrm{EEA}}$ (baseline), $g_{\mathrm{EEA}}J_p$ (within a selective pool) and $g_{\mathrm{EEA}}J_m$ (between selective pools); recurrent NMDA is pooled analogously. $E -> I$, $I -> E$ and $I -> I$ connections are all-to-all with weights $g_{\mathrm{EIA}}$, $g_{\mathrm{IE}}$ and $g_{\mathrm{II}}$ respectively. Base conductances ([](#tab-cond)) are calibrated for $N=2000$ and rescaled by $1600/N_E$ (excitatory) and $400/N_I$ (inhibitory) for other network sizes.

### S1.4 Background input and stimulation protocol

Every neuron receives $N_{\mathrm{ext}}=800$ independent external AMPA synapses driven by Poisson spike trains at $3\ \mathrm{Hz}$ each ($2.4\ \mathrm{kHz}$ aggregate), reproducing cortical spontaneous activity. A $200\ \mathrm{ms}$ spontaneous period is followed by a cue of $200\ \mathrm{Hz}$ applied to $S_1$ over $t\in[200,700]\ \mathrm{ms}$, then a delay period $[700,1200]\ \mathrm{ms}$. An optional distractor is applied to $S_2$ over $[800,1100]\ \mathrm{ms}$ at either $0$ or $200\ \mathrm{Hz}$. Total simulation time is $1200\ \mathrm{ms}$. Equations were integrated with the forward Euler method at $dt=0.02\ \mathrm{ms}$; independent trials differ only in random seed.

### S1.5 Modelled regimes

NMDA and GABA conductances are scaled multiplicatively relative to control. The three main-text regimes hold $g_{\mathrm{GABA}}$ fixed and vary $g_{\mathrm{NMDA}}$: control ($1.00$), SCZ-like ($0.95$, $-5\%$) and OCD-like ($1.10$, $+10\%$). The effective recurrent NMDA conductance is therefore $g_{\mathrm{EEN}}\times\{0.95,1.00,1.10\}=\{0.157,0.165,0.182\}\ \mathrm{nS}$.

```{table} Single-neuron parameters (excitatory / inhibitory).
:label: tab-neuron
| Parameter | Excitatory | Inhibitory |
|---|---|---|
| Resting potential $E_L$ | $-70\ \mathrm{mV}$ | $-70\ \mathrm{mV}$ |
| Threshold $V_{\mathrm{th}}$ | $-50\ \mathrm{mV}$ | $-50\ \mathrm{mV}$ |
| Reset $V_{r}$ | $-55\ \mathrm{mV}$ | $-55\ \mathrm{mV}$ |
| Membrane capacitance $C_m$ | $0.5\ \mathrm{nF}$ | $0.2\ \mathrm{nF}$ |
| Leak conductance $g_L$ | $25\ \mathrm{nS}$ | $20\ \mathrm{nS}$ |
| Membrane time constant $\tau_m=C_m/g_L$ | $20\ \mathrm{ms}$ | $10\ \mathrm{ms}$ |
| Refractory period $t_{\mathrm{ref}}$ | $2\ \mathrm{ms}$ | $1\ \mathrm{ms}$ |
```

```{table} Synaptic kinetics and reversal potentials.
:label: tab-syn
| Parameter | Value | Parameter | Value |
|---|---|---|---|
| $\tau_{\mathrm{AMPA}}$ | $2\ \mathrm{ms}$ | $V_E$ | $0\ \mathrm{mV}$ |
| $\tau_{\mathrm{NMDA,rise}}$ | $2\ \mathrm{ms}$ | $V_I$ | $-70\ \mathrm{mV}$ |
| $\tau_{\mathrm{NMDA,decay}}$ | $100\ \mathrm{ms}$ | $[\mathrm{Mg}^{2+}]$ | $1\ \mathrm{mM}$ |
| $\tau_{\mathrm{GABA}}$ | $10\ \mathrm{ms}$ | Synaptic delay | $0.5\ \mathrm{ms}$ |
| $\alpha$ (NMDA) | $0.5\ \mathrm{ms^{-1}}$ | Integration step $dt$ | $0.02\ \mathrm{ms}$ |
```

```{table} Base synaptic conductances (calibrated for $N=2000$; the excitatory/inhibitory scale factors equal $1$ at this size).
:label: tab-cond
| Conductance | Value | Conductance | Value |
|---|---|---|---|
| $g_{\mathrm{ext},E}$ (external AMPA, $E$) | $2.10\ \mathrm{nS}$ | $g_{\mathrm{EEN}}$ (rec. NMDA, $E -> E$) | $0.165\ \mathrm{nS}$ |
| $g_{\mathrm{ext},I}$ (external AMPA, $I$) | $1.62\ \mathrm{nS}$ | $g_{\mathrm{EIN}}$ (rec. NMDA, $E -> I$) | $0.130\ \mathrm{nS}$ |
| $g_{\mathrm{EEA}}$ (rec. AMPA, $E -> E$) | $0.050\ \mathrm{nS}$ | $g_{\mathrm{IE}}$ (GABA, $I -> E$) | $1.30\ \mathrm{nS}$ |
| $g_{\mathrm{EIA}}$ (rec. AMPA, $E -> I$) | $0.040\ \mathrm{nS}$ | $g_{\mathrm{II}}$ (GABA, $I -> I$) | $1.00\ \mathrm{nS}$ |
```

```{table} Network architecture and stimulation protocol.
:label: tab-arch
| Quantity | Value | Quantity | Value |
|---|---|---|---|
| Total neurons $N$ | $2000$ | Cue (to $S_1$) | $200\ \mathrm{Hz}$, $[200,700]\ \mathrm{ms}$ |
| Excitatory $N_E$ / Inhibitory $N_I$ | $1600 / 400$ | Delay period | $[700,1200]\ \mathrm{ms}$ |
| Selective pools $\mathrm{S1}=\mathrm{S2}$ | $240$ each | Distractor (to $S_2$) | $0$ or $200\ \mathrm{Hz}$, $[800,1100]\ \mathrm{ms}$ |
| Non-selective pool $\mathrm{NS}$ | $1120$ | Total runtime | $1200\ \mathrm{ms}$ |
| External synapses $N_{\mathrm{ext}}$ | $800$ at $3\ \mathrm{Hz}$ | Potentiation $J_p$ | $1.84$ (swept $1.75$–$1.90$) |
| Cross-pool factor $J_m$ at $J_p{=}1.84$ | $0.852$ | | |
```

(sec-lambda)=
## S2. The $\lambda_{\mathrm{delay}}$ metric

Population firing rates were obtained from spike-count histograms smoothed with a flat (rectangular) sliding window of width $50\ \mathrm{ms}$. During the delay we form the differential rate $\Delta r(t)=r_{S_1}(t)-r_{S_2}(t)$ and fit a single exponential $\Delta r(t)=A\,e^{-\lambda t}$ to its post-peak segment by linear regression of $\log \Delta r$. The fit is evaluated over the window $[\,t_{\mathrm{off}}+100\ \mathrm{ms},\ t_{\mathrm{end}}-50\ \mathrm{ms}\,]$, starting from the peak of $\Delta r$ within that window so that the post-cue rising transient does not bias the estimate. Small $\lambda_{\mathrm{delay}}$ indicates stable maintenance; large $\lambda_{\mathrm{delay}}$ indicates rapid collapse. We define a persistence zone as $\lambda_{\mathrm{delay}}<5\ \mathrm{s^{-1}}$ (half-life $\approx 139\ \mathrm{ms}$); equivalently $1/\lambda_{\mathrm{delay}}>0.2\ \mathrm{s}$. Networks that never encode the cue or that enter a runaway state return an undefined ($\mathrm{NaN}$) value and are treated as a separate failure category rather than as $\lambda=0$.

(sec-stats)=
## S3. Multi-seed statistical protocol

To characterise stability beyond single realisations, each parameter combination was simulated with $19$ independent random seeds. The analysis grid crossed six NMDA/GABA conductance settings — $(g_{\mathrm{NMDA}}/g_{\mathrm{GABA}})\in\{(0.8/0.4),(0.9/0.3),(1.0/0.3),(1.0/0.6),(1.1/0.4),(1.1/0.6)\}$ — with two recurrent weights ($J_p=1.75$ and $1.88$), run separately with and without the $S_2$ distractor. Because per-condition $\lambda_{\mathrm{delay}}$ distributions departed from normality (Shapiro–Wilk), non-parametric tests were used throughout: Kruskal–Wallis across conditions (with $\eta^2$ effect size), pairwise Mann–Whitney $U$ with Bonferroni correction, and Spearman rank correlations between $\lambda_{\mathrm{delay}}$ and other delay-period metrics. [](#tab-desc)–[](#tab-spear) report the with-distractor analysis used for the main-text statistics.

```{figure} FigureS1_NMDA_sweep.png
:label: fig-nmda-sweep
:width: 82%
:align: center

**NMDA-conductance sweep at fixed GABA ($g_{\mathrm{GABA}}=1.0$).** $\lambda_{\mathrm{delay}}$ as a function of the NMDA conductance multiplier for three recurrent weights $J_p$. The shaded band is the persistence zone ($\lambda_{\mathrm{delay}}<5$); dotted reference lines mark the three main-text regimes (SCZ $0.95$, control $1.00$, OCD $1.10$). Increasing NMDA drives the network from non-encoding/collapse at low values, through a high-$\lambda$ unstable band, into deep persistence at $g_{\mathrm{NMDA}}\geq 1.0$; the transition shifts to lower NMDA as $J_p$ increases. Values below $g_{\mathrm{NMDA}}\approx0.7$ are omitted because the network fails to encode the cue and $\lambda_{\mathrm{delay}}$ is undefined. Generated from the single-seed grid sweep (`Grid_results_200Hz.csv`).
```

(sec-tables)=
## S4. Statistical tables

```{table} Descriptive statistics of $\lambda_{\mathrm{delay}}$ (s$^{-1}$) in the persistent regime, seed-level means, with distractor. $n$ is the number of seeds (of $19$) that reached the persistent regime for that cell.
:label: tab-desc
| $J_p$ | NMDA | GABA | $n$ | Mean | SD | Median | [Min, Max] |
|---|---|---|---|---|---|---|---|
| 1.75 | 1.00 | 0.60 | 11 | 32.54 | 1.82 | 32.20 | [30.28, 37.22] |
| 1.75 | 1.10 | 0.40 | 19 | 33.99 | 3.46 | 34.15 | [27.65, 40.83] |
| 1.75 | 1.10 | 0.60 | 19 | 13.64 | 11.85 | 7.23 | [2.26, 34.14] |
| 1.88 | 0.80 | 0.40 | 3 | 33.90 | 0.61 | 34.15 | [33.21, 34.34] |
| 1.88 | 0.90 | 0.30 | 19 | 34.50 | 2.49 | 34.36 | [29.53, 41.16] |
| 1.88 | 1.00 | 0.30 | 19 | 4.60 | 8.06 | 1.65 | [0.60, 28.12] |
| 1.88 | 1.00 | 0.60 | 19 | 0.92 | 1.13 | 0.61 | [−0.08, 4.97] |
| 1.88 | 1.10 | 0.40 | 19 | 0.31 | 0.39 | 0.27 | [−0.33, 1.21] |
| 1.88 | 1.10 | 0.60 | 19 | 0.26 | 0.34 | 0.24 | [−0.25, 1.16] |
```

```{table} Omnibus Kruskal–Wallis tests on $\lambda_{\mathrm{delay}}$ across conditions (with distractor).
:label: tab-kw
| Grouping | $H$ (df) | $p$ | $\eta^2$ |
|---|---|---|---|
| $J_p=1.75$ | $25.97\ (2)$ | $2.30\times10^{-6}$ | $0.521$ |
| $J_p=1.88$ | $75.87\ (5)$ | $6.14\times10^{-15}$ | $0.770$ |
| Combined | $40.88\ (5)$ | $9.92\times10^{-8}$ | $0.254$ |
```

```{table} Overall persistent vs. transient comparison (with distractor): median $\lambda_{\mathrm{delay}}$ and Mann–Whitney $U$.
:label: tab-pt
| Regime | $n$ | Median $\lambda_{\mathrm{delay}}$ | Mann–Whitney |
|---|---|---|---|
| Persistent | 155 | 3.33 | $U=4515,\ p=5.6\times10^{-5}$ |
| Transient | 85 | 29.21 | |
```

```{table} Spearman rank correlations between $\lambda_{\mathrm{delay}}$ and delay-period metrics (persistent regime, with distractor). Negative values indicate that faster decay accompanies weaker maintenance.
:label: tab-spear
| Metric | $\rho$ | $p$ |
|---|---|---|
| Attractor robustness ($\min \Delta r$) | $-0.952$ | $<10^{-80}$ |
| Signal-to-noise ratio (delay) | $-0.947$ | $<10^{-76}$ |
| Firing-rate variability $\sigma$ | $+0.921$ | $<10^{-63}$ |
| Minimum $S_1$ rate (delay) | $-0.910$ | $<10^{-59}$ |
| Mean $S_1$ rate (delay) | $-0.838$ | $<10^{-41}$ |
| Delay gain | $-0.613$ | $<10^{-16}$ |
```

:::{note}
$\lambda_{\mathrm{delay}}$, attractor robustness ($\min\Delta r$) and several of the metrics in [](#tab-spear) are derived from the same $\Delta r(t)$ trajectory; the correlations therefore quantify internal consistency of the metric rather than independent validation. All values were produced by the accompanying code (`Working_Memory_Model.py`, `Run_Grid_Trials_points.py`, `statistical_analysis.py`).
:::
