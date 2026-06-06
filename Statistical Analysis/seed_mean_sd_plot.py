"""
Mean ± SD across random seeds — Reviewer-requested plot
Pipeline:
  1. Filter persistent regime
  2. Mean across sub-runs within each (Jp, nmda, gaba, seed)  → seed-level values
  3. Mean ± SD across the 19 seeds per condition              → plot points

Conditions mapped to schizophrenia symptom categories per Loh, Rolls & Deco (2007)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# ── Theme ──────────────────────────────────────────────────────────────────────
BG      = 'white'
PANEL   = 'white'
TXT     = 'black'
GRID_C  = '#dddddd'
SPINE_C = '#aaaaaa'
TICK_C  = 'black'
ANNOT_C = '#444444'
LEG_BG  = 'white'
LEG_EDG = '#aaaaaa'

# ── Load & aggregate ──────────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/points_results_1.csv')

pers = df[df['regime'] == 'persistent'].copy()

seed_level = (
    pers.groupby(['Jp', 'nmda_scale', 'gaba_scale', 'seed'])
    .mean(numeric_only=True)
    .reset_index()
)

METRICS = {
    'snr_delay':           'SNR — delay period\n(working memory fidelity)',
    'r1_delay_mean':       'Mean S1 firing rate\nduring delay (Hz)',
    'attractor_robustness':'Attractor robustness\n(basin depth proxy)',
    'r1_delay_std':        'Firing-rate variability σ\n(noise / instability)',
    'delay_gain':          'Delay gain\n(persistent vs. spontaneous)',
    'persistent':          'Fraction persistent\ntrials',
}

def make_cond_stats(seed_level, metric):
    result = (
        seed_level.groupby(['Jp', 'nmda_scale', 'gaba_scale'])[metric]
        .agg(['mean', 'std', 'sem', 'count'])
        .reset_index()
    )
    result.columns = ['Jp', 'nmda_scale', 'gaba_scale', 'mean', 'sd', 'sem', 'n']
    return result

cond_stats = {metric: make_cond_stats(seed_level, metric) for metric in METRICS}

# ── Condition metadata ────────────────────────────────────────────────────────
COND_META = {
    (0.8, 0.4): dict(label='NMDA↓\n(0.8 / 0.4)',       category='Cognitive &\nNegative Sx',
                     color='#D62728', marker='v', ls='--'),
    (0.9, 0.3): dict(label='NMDA↓+GABA↓\n(0.9 / 0.3)', category='Positive Sx',
                     color='#FF7F0E', marker='D', ls='-.'),
    (1.0, 0.3): dict(label='NMDA=1 GABA↓\n(1.0 / 0.3)', category='Intermediate',
                     color='#7F7F7F', marker='s', ls=':'),
    (1.0, 0.6): dict(label='Normal\n(1.0 / 0.6)',        category='Normal',
                     color='#1F77B4', marker='o', ls='-'),
    (1.1, 0.4): dict(label='NMDA↑\n(1.1 / 0.4)',        category='Enhanced\nstability',
                     color='#2CA02C', marker='^', ls='-'),
    (1.1, 0.6): dict(label='NMDA↑+GABA↑\n(1.1 / 0.6)', category='Max\nstability',
                     color='#9467BD', marker='*', ls='-'),
}

JP_VALS  = sorted(seed_level['Jp'].unique())
JP_ALPHA = {1.75: 0.55, 1.88: 1.0}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
axes = axes.flatten()
fig.patch.set_facecolor(BG)

x_positions = np.arange(len(COND_META))

for ax_idx, (metric, ylabel) in enumerate(METRICS.items()):
    ax = axes[ax_idx]
    ax.set_facecolor(PANEL)
    ax.grid(axis='y', color=GRID_C, linewidth=0.7, zorder=0)
    ax.grid(axis='x', color=GRID_C, linewidth=0.4, linestyle=':', zorder=0)

    cond_keys = list(COND_META.keys())
    df_m = cond_stats[metric]

    for jp_idx, jp in enumerate(JP_VALS):
        sub = df_m[df_m['Jp'] == jp]
        jitter = (jp_idx - 0.5) * 0.18

        means, sds, xs, colors, markers, labels = [], [], [], [], [], []
        for xi, (nmda, gaba) in enumerate(cond_keys):
            row = sub[(sub['nmda_scale'] == nmda) & (sub['gaba_scale'] == gaba)]
            if row.empty:
                means.append(np.nan); sds.append(np.nan)
            else:
                means.append(float(row['mean'].iloc[0]))
                sds.append(float(row['sd'].iloc[0]))
            meta = COND_META[(nmda, gaba)]
            xs.append(xi + jitter)
            colors.append(meta['color'])
            markers.append(meta['marker'])
            labels.append(meta['label'])

        means = np.array(means, dtype=float)
        sds   = np.array(sds,   dtype=float)

        # connecting line
        valid = ~np.isnan(means)
        ax.plot(np.array(xs)[valid], means[valid],
                color='#555555', linewidth=0.9, alpha=JP_ALPHA[jp] * 0.6,
                linestyle='--', zorder=2)

        # error bars + markers
        for xi2, (x_pos, m, sd, col, mk) in enumerate(zip(xs, means, sds, colors, markers)):
            if np.isnan(m):
                continue
            ax.errorbar(x_pos, m, yerr=sd,
                        fmt='none', ecolor=col, elinewidth=2.2,
                        capsize=6, capthick=2.2,
                        alpha=JP_ALPHA[jp], zorder=3)
            ms = 11 if mk == '*' else 9
            ax.plot(x_pos, m, marker=mk, color=col,
                    markersize=ms,
                    markeredgecolor='black',
                    markeredgewidth=0.8 if jp == 1.75 else 1.8,
                    alpha=JP_ALPHA[jp], zorder=4, linestyle='none')

            # annotate n= once
            if jp_idx == 0:
                nmda_k, gaba_k = cond_keys[xi2]
                row2 = df_m[(df_m['Jp'] == jp) &
                            (np.isclose(df_m['nmda_scale'], nmda_k)) &
                            (np.isclose(df_m['gaba_scale'],  gaba_k))]
                if not row2.empty:
                    n = int(row2['n'].iloc[0])
                    ymin = ax.get_ylim()[0]
                    ax.text(x_pos, ymin, f'n={n}', ha='center', va='top',
                            fontsize=6.5, color=ANNOT_C, zorder=5)

    # x-axis labels
    tick_labels = [COND_META[k]['label'] for k in cond_keys]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=8.5, color=TICK_C)
    ax.set_xlim(-0.6, len(cond_keys) - 0.4)

    # y-axis
    ax.set_ylabel(ylabel, color=TXT, fontsize=9.5, labelpad=6)
    ax.tick_params(axis='y', colors=TICK_C, labelsize=8.5)
    ax.tick_params(axis='x', colors=TICK_C, length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_C)

    # panel title
    ax.set_title(ylabel.replace('\n', ' '),
                 color=TXT, fontsize=10, fontweight='bold', pad=10)

    # subtle condition background band
    for xi, (nmda, gaba) in enumerate(cond_keys):
        ax.axvspan(xi - 0.5, xi + 0.5,
                   color=COND_META[(nmda, gaba)]['color'],
                   alpha=0.05, zorder=0)

# ── Legend ────────────────────────────────────────────────────────────────────
cond_handles = [
    Line2D([0], [0],
           marker=COND_META[k]['marker'],
           color='none',
           markerfacecolor=COND_META[k]['color'],
           markeredgecolor='black',
           markersize=9 if COND_META[k]['marker'] != '*' else 12,
           label=COND_META[k]['label'].replace('\n', ' '))
    for k in COND_META
]

jp_handles = [
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor='#555555',
           markeredgecolor='black',
           markersize=9,
           markeredgewidth=0.8 if jp == 1.75 else 1.8,
           alpha=JP_ALPHA[jp],
           label=f'Jp = {jp}  ({"thin" if jp==1.75 else "thick"} edge)')
    for jp in JP_VALS
]

eb_handle = Line2D([0], [0], color='#555555', linewidth=2,
                   label='Error bar = ±1 SD across seeds')

leg1 = fig.legend(handles=cond_handles, title='Condition (NMDA / GABA)',
                  title_fontsize=9, fontsize=8.5,
                  loc='lower left', bbox_to_anchor=(0.01, 0.01),
                  ncol=3, framealpha=1.0,
                  facecolor=LEG_BG, edgecolor=LEG_EDG,
                  labelcolor=TXT)
leg1.get_title().set_color(TXT)

leg2 = fig.legend(handles=jp_handles + [eb_handle],
                  title='Jp / error bars',
                  title_fontsize=9, fontsize=8.5,
                  loc='lower right', bbox_to_anchor=(0.99, 0.01),
                  ncol=1, framealpha=1.0,
                  facecolor=LEG_BG, edgecolor=LEG_EDG,
                  labelcolor=TXT)
leg2.get_title().set_color(TXT)

# ── Global title ──────────────────────────────────────────────────────────────
fig.suptitle(
    'Attractor Network Simulations — Mean ± SD across random seeds\n'
    'Each point: mean over 19 seeds  |  Error bars: ±1 SD  |  Persistent regime only\n'
    'Loh, Rolls & Deco (2007) framework — Schizophrenia / OCD computational model',
    color=TXT, fontsize=12, fontweight='bold', y=0.995, va='top')

plt.tight_layout(rect=[0, 0.11, 1, 0.97])

out = '/mnt/user-data/outputs/mean_sd_across_seeds.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
print(f'Saved → {out}')

# ── Print the aggregated table ────────────────────────────────────────────────
print('\n=== SEED-AGGREGATED TABLE (mean ± SD, n=19 seeds per condition) ===\n')
for metric in ['snr_delay', 'r1_delay_mean', 'attractor_robustness']:
    df_m = cond_stats[metric]
    print(f'  {metric}:')
    print(f"  {'Jp':>6}  {'NMDA':>6}  {'GABA':>6}  {'n':>4}  {'Mean':>10}  {'SD':>10}")
    print('  ' + '-'*52)
    for _, row in df_m.sort_values(['Jp','nmda_scale','gaba_scale']).iterrows():
        print(f"  {row['Jp']:>6.2f}  {row['nmda_scale']:>6.2f}  {row['gaba_scale']:>6.2f}  "
              f"{int(row['n']):>4}  {row['mean']:>10.3f}  {row['sd']:>10.3f}")
    print()
