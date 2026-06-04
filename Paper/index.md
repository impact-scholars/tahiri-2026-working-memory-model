---
title: "NMDA Modulates Working Memory Attractor Stability: Opposite Regimes Can Produce Schizophrenia-like Instability and OCD-like Overstability"
exports:
  - format: typst
abstract: |
    Working memory (WM) depends on the sustained, selective firing of prefrontal neurons during a delay
    period — a property implemented in computational models as a stable attractor state. NMDA receptors
    are central to the excitatory recurrent currents that maintain these attractors. Using a biologically
    plausible network with AMPA, NMDA, and GABA$_A$ synaptic currents, we show that NMDA conductance may
    act as a bidirectional gain control on attractor depth. To quantify this, we introduce
    $\lambda_{\mathrm{delay}}$, an exponential decay constant of the selective population firing rate
    during the delay period, where low values reflect stable memory maintenance and high values reflect
    attractor collapse. A 5% reduction in NMDA (schizophrenia-like) destabilizes the delay-period
    attractor, causing memory collapse and high $\lambda_{\mathrm{delay}}$ values across the full
    recurrent weight range. Conversely, a 10% increase in NMDA (OCD-like) over-stabilizes the attractor,
    locking activity into persistence even at low synaptic weights. These opposite failure modes,
    instability and overstability, emerge from the same parameter and suggest a possible mechanistic
    axis that could link two clinically distinct psychiatric conditions.

acknowledgments: |
    This work was supported by the Impact Scholars Program 2025. The first author is supported by the CNRST PASS doctoral scholarship program. The authors thank Archishman Biswas for his technical support with the simulation code, helpful discussions, and guidance that greatly contributed to the development of this work. The authors used AI assistance (Claude, Anthropic) to assist with scientific writing and code review during the preparation of this manuscript. All scientific content, results, and conclusions were verified and approved by the authors.

    **Correspondence:** Tahiri, H. (tahirihamza573@gmail.com) and Kumar, A. (arvind.k.panchal@gmail.com)
---


# Description

Working memory (WM) is the ability to hold information in mind over a brief delay in the absence of
sensory input. At its neural substrate lies a prefrontal cortical circuit capable of sustaining elevated, selective firing
rates long after a cue has disappeared. Attractor network model captures this property. Recurrent excitatory connections encodes this by creating stable , self-reinforcing activity states (attractors) {cite:p}`brunel2001effects,compte_synaptic_2000,wang1999synaptic`.
The depth of these attractor basins determines how robustly a memory is maintained against the inevitable statistical fluctuations
of stochastic spiking, the neural "noise" that constantly perturbs the system.

NMDA receptors sustain WM attractors, their slow kinetics and voltage
dependency provide the persistence firing rates of the selective pool
of neurons during the delay period {cite:p}`wang2001synaptic`. Schizophrenia is associated with NMDA receptor hypofunction {cite:p}`coyle2003converging`. While obsessive-compulsive disorder (OCD) is associated with hyperactive, perseverative
prefrontal states that resist updating {cite:p}`milad2012obsessive`. A unified computational account
of how these opposite deviations produce opposite WM failure modes has not been fully formalized.

We implemented a biophysically realistic spiking network based on the
{cite:p}`brunel2001effects,loh2007dynamical` framework. The network consists of:

- Two selective excitatory pools (S1, S2) representing competing memory items.
- One non-selective excitatory pool (NS) providing background excitatory noise
- One global inhibitory interneuron pool of GABAergic neurons providing feedback inhibition
- Background Poisson inputs per neuron, generating realistic stochastic fluctuations
- Three synaptic receptor types: AMPA (fast excitation), NMDA (slow excitation, Mg²⁺-gated),
  and GABA$_A$ (inhibition)

Within-pool recurrent weights were set stronger than between-pool weights, this
creates a winner-take-all dynamics, a necessary feature for selective memory encoding. A 200ms spontaneous state simulation followed by an external cue stimulation applied to pool S1 for a 500 ms encoding window ($t_{\mathrm{on}}$ = 200 ms,
$t_{\mathrm{off}}$ = 700 ms), followed by a delay period during which the network maintained or
failed to maintain the encoded memory without external support. A distractor cue was simultaneously
applied to pool S2 ($t_{\mathrm{on}}$ = 800 ms, $t_{\mathrm{off}}$ = 1100 ms). Total simulation
duration was 1200 ms. All simulations were implemented in Brian2 {cite}`stimberg2019brian2`, a
Python-based spiking neural network simulator. Population firing rates were smoothed with a Gaussian
kernel ($\sigma$ = 50 ms), applied to the raw spike-count histogram of each pool, before fitting
the exponential decay function.

To quantify WM maintenance beyond a binary persistent/non-persistent classification, we introduced
$\lambda_{\mathrm{delay}}$, the exponential decay constant fitted to the smoothed S1 population
rate during the delay period. During the delay, we compute the differential firing rate 
$$\Delta r(t) = r_{S1}(t) - r_{S2}(t)$$
and fit an exponential function 
$$\Delta r(t) = A \, e^{-\lambda t}$$
to the post-peak trajectory. The fit is applied for [$t_{\mathrm{off}} + 100 ms, $t_{\mathrm{end}} − 50 ms] (to avoid the
cue-offset transient) to the end of the simulation. 

A small $\lambda_{\mathrm{delay}}$ indicates that S1 activity remains
elevated and stable throughout the delay (strong persistence), while a large $\lambda_{\mathrm{delay}}$
indicates rapid collapse of activity toward baseline. We defined a persistence zone as
$\lambda_{\mathrm{delay}} < 5$, empirically corresponding to trials in which the network
successfully sustained a memory representation until the end of the simulation window. Trials with
$\lambda_{\mathrm{delay}} > 5$  (half-life $\approx$ 139 ms) had S1 firing return to
within 3 Hz of baseline before simulation end, indicating collapse; trials with
$\lambda_{\mathrm{delay}} < 5$ sustained S1 activity more than 10 Hz above baseline for the full
delay window. Equivalently, one may interpret $1/\lambda_{\mathrm{delay}}$ as a memory persistence
timescale: larger values indicate longer-lasting, more stable representations, and the persistence
zone corresponds to $1/\lambda_{\mathrm{delay}} > 0.2$ s.

NMDA conductance was scaled relative to the control condition ($nmda_{\mathrm{scale}} = 1.0$)
to model three regimes:


| Condition | $g_{\mathrm{NMDA}}$ | $g_{\mathrm{GABA}}$ | Interpretation |
|-----------|:-------------------:|:-------------------:|----------------|
| **Control** | 1.00 | 1.00 | Healthy baseline |
| **SCZ (−NMDA)** | 0.95 | 1.00 | −5% NMDA (receptor hypofunction) |
| **OCD (+NMDA)** | 1.10 | 1.00 | +10% NMDA (receptor hyperfunction) |

These perturbations follow established theoretical frameworks: reduced NMDA destabilizes prefrontal working memory attractors in SCZ {cite:p}`loh2007dynamical, rolls2008computational`, whereas increased glutamatergic drive over-deepens attractor basins in OCD {cite:p}`rolls2008attractor`. Biologically, the −5% reduction reflects moderate post-mortem NMDA-receptor hypofunction estimates in SCZ {cite:p}`catts2016quantitative`; a conservative value was chosen to avoid a global excitability collapse. The +10% increase reflects glutamatergic overactivity reported in OCD {cite:p}`pittenger2011glutamate`. Because the OCD direction is less precisely constrained experimentally, the larger value serves as an illustrative operating point. Both values define illustrative regimes rather than exact disease-calibrated magnitudes; as Figure S1 demonstrates, qualitative transitions remain robust across the perturbation range.

```{figure} Fig1.png
:name: figure-main
:alt: NMDA bidirectionally regulates the recurrent coupling threshold for persistent working memory activity.


**A--C.** Representative firing-rate traces (firing rate in Hz, y-axis) of the memory-selective
population (S1), competing selective population (S2), non-selective (NS) and inhibitory neurons during simulations under three NMDA scaling regimes.

**(A)** Control NMDA ($nmda_{\mathrm{scale}} = 1.0$): Healthy regime.

**(B)** OCD-like ($nmda_{\mathrm{scale}} = 1.1$): Receptor hyperfunction regime.

**(C)** SCZ-like ($nmda_{\mathrm{scale}} = 0.95$): Receptor hypofunction regime.
\
**D.** $\lambda_{\mathrm{delay}}$ as a function of recurrent synaptic weight $J_p$ across the
three NMDA conditions. The shaded region marks the persistence 
zone ($\lambda_{\mathrm{delay}} < 5$). Conditions are distinguished by line style as well as color
to ensure accessibility.
```

```{note} How to read $\lambda_{\mathrm{delay}}$ in the figure
**Figure 1-D** of @figure-main is the key diagnostic plot. Each curve shows how
$\lambda_{\mathrm{delay}}$ changes as the recurrent synaptic weight $J_p$ increases (x-axis).
**Curves that stay above the dashed threshold** ($\lambda_{\mathrm{delay}} = 5$) represent conditions where the
network consistently *fails* to maintain working memory — high decay, fast collapse.
**Curves that drop into the shaded persistence zone** represent conditions where memory is
robustly held. The steepness of the drop tells you how sensitive that regime is to changes in $J_p$:
a sharp drop (OCD-like) means memory locks in easily; a shallow or absent drop (SCZ-like) means the
network rarely succeeds at all.
```

Panels A–C of @figure-main illustrate single-trial population rate traces for three representative
conditions at $J_p = 1.84$. In Figure 1A (control, $\lambda_{\mathrm{delay}} = 1.751$), the S1 pool
ramps up during stimulation and sustains elevated activity at ~20 Hz throughout the delay, indicating a successful WM maintenance. The competing pools (S2, NS) remain near baseline. In Figure1B
(OCD-like, $\lambda_{\mathrm{delay}} = 1.059$), the memory attractor is even more robust: activity
is sustained at higher rates ~45 Hz, reflecting an over-deepened basin of
attraction. In contrast, Figure1C (SCZ-like, $\lambda_{\mathrm{delay}} = 28.360$) shows activity
that collapses rapidly after cue offset, returning to baseline before the end of the delay, the
attractor basin is too shallow to resist noise-driven escape.

Figure1D reveals the full picture across the $J_p$ sweep. For the **control condition**, the network
enters the persistence zone ($\lambda_{\mathrm{delay}} < 5$) at intermediate $J_p$ (~1.84-1.86),
reflecting a tuned operating point. For the **SCZ-like condition**, $\lambda_{\mathrm{delay}}$
remains elevated across nearly the entire $J_p$ range, until higher values of $J_p$ (> 1.86). For the **OCD-like condition**, the system drops into the persistence zone at
$J_p \geq 1.78$ and $\lambda_{\mathrm{delay}}$ plummets rapidly: the network locks into persistent
activity easily and at weights where the control network would still be in the non-persistent regime.

The influence of the distractor stimulus on the S2 population is modest in the current parameter
regime; stronger distractor amplitudes should be explored in future work to more directly probe
attractor competition and cross-pool interference.

These results are consistent with the hypothesis that NMDA conductance acts as a bidirectional gain
control on attractor depth, offering a possible mechanistic link between two ostensibly opposite
psychiatric presentations within a single mechanistic axis:

- In schizophrenia-like regime NMDA hypofunction flattens attractor basins, making WM representations
  fragile, easily disrupted by noise, and prone to collapse during the delay. This is consistent with the
  well-documented WM deficits and hypofrontality observed in schizophrenia
  {cite:p}`goldmanrakic1994working,rolls2007memory`, as well as the reduced signal-to-noise ratio in
  prefrontal circuits {cite:p}`winterer2004prefrontal`. Cognitive and negative symptoms, failure to
  maintain task-relevant representations and flattening of goal-directed behavior,
  could arise from this attractor instability.

- In OCD-like regime, NMDA hyperfunction may over-deepen attractor basins, causing representations to persist
  beyond their usefulness and resist updating and disengagement. The network becomes locked into its
  current attractor state, providing a hypothetical computational analog to the perseverative thoughts
  and behavioral rigidity that define OCD {cite:p}`milad2012obsessive`.

Together, these simulations suggest that the healthy prefrontal WM circuit operates in a narrow,
tuned window of NMDA conductance, close enough to the instability boundary to remain flexible and
updatable, but deep enough to maintain representations against noise. The $\lambda_{\mathrm{delay}}$
metric provides a continuous, quantitative readout of this balance, offering a potential link to
delay-period fMRI and electrophysiology signatures.

We note that the interpretation of $\lambda_{\mathrm{delay}}$ as a proxy for attractor basin depth
is a hypothesis. Alternative mechanisms, such as a change in the effective membrane time constant
of the selective population, or a shift in the competitive balance between S1 and S2, could produce
similar changes in the decay timescale without necessarily reshaping the attractor landscape.
Disambiguating these alternatives would require explicit computation of the flow field, as in
{cite:p}`loh2007dynamical`, and is left for future work.

```{figure} stq.png
:name: figure-stats
:alt: Statistical analysis of lambda_delay across conditions showing mean SD distributions and correlation with attractor robustness.

**A.** Mean ± SD of $\lambda_{\mathrm{delay}}$ per condition across seeds 
for $J_p = 1.75$ (faded) and $J_p = 1.88$ (solid). Conditions differ 
significantly across the parameter space (Kruskal-Wallis: $H(5) = 40.9$, 
$p = 9.92 \times 10^{-8}$, $\eta^2 = 0.254$).

**B.** Violin plots showing the full distribution of $\lambda_{\mathrm{delay}}$ 
in the persistent regime across all seeds and $J_p$ values per condition. 
Black bars indicate the median.

**C.** Scatter plot of $\lambda_{\mathrm{delay}}$ versus attractor robustness 
showing a strong Spearman correlation ($\rho = -0.952$, $p < 0.001$).
```

To characterise attractor stability statistically across seeds and 
conditions, we tested different combinations of NMDA and GABA conductances 
at two recurrent weight values ($J_p = 1.75$ and $J_p = 1.88$), running 
multiple trials with different random seeds and computing 
$\lambda_{\mathrm{delay}}$ for each trial. The persistent regime ($n = 155$) 
showed a median $\lambda_{\mathrm{delay}}$ of 3.3 compared to 29.2 in the 
transient regime, a difference validated by a Mann-Whitney test 
($U = 4515$, $p < 0.001$). A Kruskal-Wallis test confirmed significant 
differences across conditions ($H(5) = 40.9$, $p = 9.92 \times 10^{-8}$, 
$\eta^2 = 0.254$), establishing that $\lambda_{\mathrm{delay}}$ is sensitive 
to changes in NMDA/GABA conductance across the parameter space. 
$\lambda_{\mathrm{delay}}$ showed a strong Spearman correlation with 
attractor robustness ($\rho = -0.952$, $p < 0.001$), supporting its 
validity as a continuous measure of working memory attractor stability, 
capturing the same information as binary persistence classification but 
with greater resolution.

This work demonstrates that a single synaptic parameter, NMDA receptor conductance, is sufficient
to drive a prefrontal attractor network between three qualitatively distinct regimes: healthy WM
maintenance, schizophrenia-like memory collapse, and OCD-like pathological persistence. The
$\lambda_{\mathrm{delay}}$ metric provides a graded, continuous measure of this axis, moving beyond
binary persistent/non-persistent classifications and offering a quantitative bridge between in silico
dynamics and delay-period neural signatures measurable with fMRI or multi-unit electrophysiology.
Future work should examine how dopaminergic D1 receptor modulation interacts with NMDA-mediated
stability, and whether combined NMDA/GABA perturbations can reproduce the positive symptoms of
schizophrenia, including spontaneous wandering between attractor states,
as predicted by prior theoretical work {cite:p}`deco2007extended,loh2007dynamical`. 

