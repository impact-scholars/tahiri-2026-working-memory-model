"""
Statistical Analysis of Attractor Network Simulation Results
Schizophrenia / OCD Computational Model
Based on: Loh, Rolls & Deco (2007) - A Dynamical Systems Hypothesis of Schizophrenia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, shapiro
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Theme ──────────────────────────────────────────────────────────────────────
BG      = 'white'
PANEL   = 'white'
TXT     = 'black'
GRID_C  = '#dddddd'
SPINE_C = '#aaaaaa'
TICK_C  = 'black'
ANNOT_C = '#222222'
LEG_BG  = 'white'
LEG_EDG = '#aaaaaa'

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH  = 'points_results_1.csv'
OUT_PLOT   = 'statistical_analysis.png'
OUT_REPORT = 'statistical_report.txt'

# ─── LOAD & LABEL ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

def label_condition(row):
    n, g = row['nmda_scale'], row['gaba_scale']
    if   n == 1.0 and g == 0.6: return 'point 1'
    elif n == 0.8 and g == 0.4: return 'point 2'
    elif n == 0.9 and g == 0.3: return 'point 3'
    elif n == 1.1 and g == 0.4: return 'point 4'
    elif n == 1.1 and g == 0.6: return 'point 5'
    elif n == 1.0 and g == 0.3: return 'point 6'
    return f'NMDA={n}, GABA={g}'

df['condition'] = df.apply(label_condition, axis=1)

METRICS = {
    'snr_delay':           'SNR during delay (working memory fidelity)',
    'attractor_robustness':'Attractor robustness',
    'r1_delay_mean':       'Mean S1 firing rate – delay period (Hz)',
    'r1_delay_std':        'Firing-rate variability – delay (Hz)',
    'delay_gain':          'Delay gain (persistent vs. spontaneous)',
    'persistent':          'Fraction persistent trials',
    'survives_distractor': 'Distractor survival rate',
}

conditions = sorted(df['condition'].unique())

# Use matplotlib tab10 — works well on white background
cmap = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
color_map = {c: cmap[i] for i, c in enumerate(conditions)}

# ── Axes style helper ─────────────────────────────────────────────────────────
def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TXT, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=TXT, fontsize=9)
    ax.set_ylabel(ylabel, color=TXT, fontsize=9)
    ax.tick_params(axis='both', colors=TICK_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_C)
    ax.grid(axis='y', color=GRID_C, linewidth=0.5, linestyle='--')

# ── Post-hoc pairwise Mann-Whitney + Bonferroni ───────────────────────────────
def posthoc_mw(data, metric, groups):
    pairs = list(combinations(groups, 2))
    results = []
    for g1, g2 in pairs:
        a = data[data['condition'] == g1][metric].dropna()
        b = data[data['condition'] == g2][metric].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        r = 1 - 2*stat/(len(a)*len(b))
        results.append({'G1': g1, 'G2': g2, 'U': stat, 'p_raw': p, 'r': r})
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    out['p_bonf'] = np.minimum(out['p_raw'] * len(out), 1.0)
    out['sig'] = out['p_bonf'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))
    return out

# ─── TEXT REPORT ─────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 78)
lines.append("STATISTICAL ANALYSIS REPORT")
lines.append("Attractor Network Simulations — Schizophrenia/OCD Computational Model")
lines.append("=" * 78)
lines.append(f"\nDataset: {len(df)} trials  |  {len(conditions)} conditions  |  "
             f"Jp values: {sorted(df['Jp'].unique())}")
lines.append(f"NMDA scales: {sorted(df['nmda_scale'].unique())}   "
             f"GABA scales: {sorted(df['gaba_scale'].unique())}\n")

lines.append("─" * 78)
lines.append("1. DESCRIPTIVE STATISTICS (per condition)")
lines.append("─" * 78)

desc_rows = []
for cond in conditions:
    sub = df[df['condition'] == cond]
    for metric in METRICS:
        col = sub[metric].astype(float)
        desc_rows.append({'Condition': cond, 'Metric': metric,
                          'N': col.count(), 'Mean': col.mean(),
                          'SD': col.std(), 'Median': col.median()})
desc_df = pd.DataFrame(desc_rows)

for metric in METRICS:
    sub = desc_df[desc_df['Metric'] == metric]
    lines.append(f"\n  {metric} — {METRICS[metric]}")
    lines.append(f"  {'Condition':<45} {'N':>4} {'Mean':>9} {'SD':>9} {'Median':>9}")
    lines.append("  " + "-" * 78)
    for _, row in sub.iterrows():
        lines.append(f"  {row['Condition']:<45} {row['N']:>4} "
                     f"{row['Mean']:>9.3f} {row['SD']:>9.3f} {row['Median']:>9.3f}")

lines.append("\n" + "─" * 78)
lines.append("2. NORMALITY CHECK (Shapiro-Wilk)")
lines.append("─" * 78)
normality_fails = 0
for cond in conditions:
    sub = df[df['condition'] == cond]
    for metric in ['snr_delay', 'attractor_robustness', 'r1_delay_mean']:
        col = sub[metric].dropna()
        if len(col) >= 3:
            stat, p = shapiro(col)
            flag = '' if p >= 0.05 else '  ← non-normal'
            if p < 0.05: normality_fails += 1
            lines.append(f"  {cond[:40]:<40}  {metric:<22} W={stat:.3f}  p={p:.4f}{flag}")
lines.append(f"\n  → {normality_fails} non-normal. Using Kruskal-Wallis + Mann-Whitney.")

lines.append("\n" + "─" * 78)
lines.append("3. KRUSKAL-WALLIS H-TEST")
lines.append("─" * 78)
lines.append(f"  {'Metric':<30} {'H':>8} {'df':>4} {'p':>10}  Sig   η²\n")

kw_results = {}
for metric in METRICS:
    groups = [df[df['condition'] == c][metric].astype(float).dropna().values
              for c in conditions if len(df[df['condition'] == c][metric].dropna()) >= 3]
    if len(groups) < 2: continue
    H, p = kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    eta2 = (H - len(groups) + 1) / (n_total - len(groups))
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    kw_results[metric] = {'H': H, 'p': p, 'eta2': eta2, 'sig': sig}
    lines.append(f"  {metric:<30} {H:>8.3f} {len(groups)-1:>4} {p:>10.4e}  {sig:<4}  {eta2:.3f}")

lines.append("\n" + "─" * 78)
lines.append("4. POST-HOC PAIRWISE (Mann-Whitney + Bonferroni)")
lines.append("─" * 78)
for metric in ['snr_delay', 'attractor_robustness', 'r1_delay_mean', 'survives_distractor']:
    ph = posthoc_mw(df, metric, conditions)
    if ph.empty: continue
    lines.append(f"\n  Metric: {metric}")
    lines.append(f"  {'Pair':<80} {'p_bonf':>10}  Sig   r")
    lines.append("  " + "-" * 78)
    for _, row in ph.iterrows():
        pair = f"{row['G1'][:35]} vs {row['G2'][:35]}"
        lines.append(f"  {pair:<80} {row['p_bonf']:>10.4f}  {row['sig']:<4}  {row['r']:.3f}")

lines.append("\n" + "─" * 78)
lines.append("5. CHI-SQUARE: PERSISTENT & SURVIVES_DISTRACTOR × CONDITION")
lines.append("─" * 78)
for col_name in ['persistent', 'survives_distractor']:
    ct = pd.crosstab(df['condition'], df[col_name].astype(int))
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    cv = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    lines.append(f"\n  {col_name}: χ²({dof})={chi2:.3f}, p={p:.4e}  {sig}  Cramér's V={cv:.3f}")
    prop = ct.iloc[:, -1] / ct.sum(axis=1)
    for cond, val in prop.items():
        lines.append(f"    {cond:<50}: {val:.3f}")

lines.append("\n" + "─" * 78)
lines.append("6. Jp EFFECT (Mann-Whitney, Jp=1.75 vs 1.88)")
lines.append("─" * 78)
for cond in conditions:
    sub = df[df['condition'] == cond]
    jp_vals = sorted(sub['Jp'].unique())
    if len(jp_vals) < 2: continue
    for metric in ['snr_delay', 'r1_delay_mean']:
        a = sub[sub['Jp'] == jp_vals[0]][metric].dropna()
        b = sub[sub['Jp'] == jp_vals[1]][metric].dropna()
        if len(a) < 3 or len(b) < 3: continue
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        lines.append(f"  {cond[:44]:<45} {metric:<20} {U:>8.1f} {p:>8.4f}  {sig}")

lines.append("\n" + "=" * 78)
lines.append("END OF REPORT")
lines.append("=" * 78)

report_text = '\n'.join(lines)
print(report_text)
with open(OUT_REPORT, 'w', encoding='utf-8') as f:
    f.write(report_text)

# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.38,
                       left=0.07, right=0.97, top=0.93, bottom=0.05)

cond_order = conditions

# ── Panel 1: SNR violin ────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
data_lists = [df[df['condition'] == c]['snr_delay'].dropna().values for c in cond_order]
parts = ax1.violinplot(data_lists, positions=range(len(cond_order)),
                       showmedians=True, showextrema=False, widths=0.7)
for pc, c in zip(parts['bodies'], cond_order):
    pc.set_facecolor(color_map[c]); pc.set_alpha(0.55)
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
for i, (vals, c) in enumerate(zip(data_lists, cond_order)):
    jitter = np.random.normal(0, 0.07, size=len(vals))
    ax1.scatter(i + jitter, vals, alpha=0.3, s=8, color=color_map[c], zorder=3)
ax1.set_xticks(range(len(cond_order)))
ax1.set_xticklabels([c.split('(')[0].strip() for c in cond_order],
                    rotation=35, ha='right', fontsize=7, color=TICK_C)
kw = kw_results.get('snr_delay', {})
if kw:
    ax1.text(0.98, 0.97, f"K-W: H={kw['H']:.1f}, p={kw['p']:.2e} {kw['sig']}",
             transform=ax1.transAxes, ha='right', va='top', fontsize=7.5,
             color='#333333', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#aaaaaa', alpha=0.8))
style_ax(ax1, 'SNR – Delay Period\n(Working Memory Fidelity)', ylabel='SNR (a.u.)')

# ── Panel 2: Attractor robustness violin ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
data_lists2 = [df[df['condition'] == c]['attractor_robustness'].dropna().values for c in cond_order]
parts2 = ax2.violinplot(data_lists2, positions=range(len(cond_order)),
                        showmedians=True, showextrema=False, widths=0.7)
for pc, c in zip(parts2['bodies'], cond_order):
    pc.set_facecolor(color_map[c]); pc.set_alpha(0.55)
parts2['cmedians'].set_color('black'); parts2['cmedians'].set_linewidth(2)
for i, (vals, c) in enumerate(zip(data_lists2, cond_order)):
    jitter = np.random.normal(0, 0.07, size=len(vals))
    ax2.scatter(i + jitter, vals, alpha=0.3, s=8, color=color_map[c], zorder=3)
ax2.axhline(0, color='#CC0000', linewidth=1, linestyle='--', alpha=0.6)
ax2.set_xticks(range(len(cond_order)))
ax2.set_xticklabels([c.split('(')[0].strip() for c in cond_order],
                    rotation=35, ha='right', fontsize=7, color=TICK_C)
kw2 = kw_results.get('attractor_robustness', {})
if kw2:
    ax2.text(0.98, 0.97, f"K-W: H={kw2['H']:.1f}, p={kw2['p']:.2e} {kw2['sig']}",
             transform=ax2.transAxes, ha='right', va='top', fontsize=7.5,
             color='#333333', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#aaaaaa', alpha=0.8))
style_ax(ax2, 'Attractor Robustness\n(Basin Depth Proxy)', ylabel='Robustness (a.u.)')

# ── Panel 3: Mean firing rate violin ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
data_lists3 = [df[df['condition'] == c]['r1_delay_mean'].dropna().values for c in cond_order]
parts3 = ax3.violinplot(data_lists3, positions=range(len(cond_order)),
                        showmedians=True, showextrema=False, widths=0.7)
for pc, c in zip(parts3['bodies'], cond_order):
    pc.set_facecolor(color_map[c]); pc.set_alpha(0.55)
parts3['cmedians'].set_color('black'); parts3['cmedians'].set_linewidth(2)
for i, (vals, c) in enumerate(zip(data_lists3, cond_order)):
    jitter = np.random.normal(0, 0.07, size=len(vals))
    ax3.scatter(i + jitter, vals, alpha=0.3, s=8, color=color_map[c], zorder=3)
ax3.set_xticks(range(len(cond_order)))
ax3.set_xticklabels([c.split('(')[0].strip() for c in cond_order],
                    rotation=35, ha='right', fontsize=7, color=TICK_C)
kw3 = kw_results.get('r1_delay_mean', {})
if kw3:
    ax3.text(0.98, 0.97, f"K-W: H={kw3['H']:.1f}, p={kw3['p']:.2e} {kw3['sig']}",
             transform=ax3.transAxes, ha='right', va='top', fontsize=7.5,
             color='#333333', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#aaaaaa', alpha=0.8))
style_ax(ax3, 'Mean S1 Firing Rate – Delay\n(Persistent Activity)', ylabel='Firing rate (Hz)')

# ── Panel 4: Persistence rate heatmap ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
pivot_pers = df.groupby(['nmda_scale', 'gaba_scale'])['persistent'].mean().unstack()
im4 = ax4.imshow(pivot_pers.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, origin='lower')
cb4 = plt.colorbar(im4, ax=ax4, label='Fraction persistent')
cb4.ax.yaxis.label.set_color(TXT); cb4.ax.tick_params(colors=TXT)
ax4.set_xticks(range(len(pivot_pers.columns)))
ax4.set_xticklabels([f'GABA={g:.1f}' for g in pivot_pers.columns], rotation=30, fontsize=8, color=TICK_C)
ax4.set_yticks(range(len(pivot_pers.index)))
ax4.set_yticklabels([f'NMDA={n:.1f}' for n in pivot_pers.index], fontsize=8, color=TICK_C)
for i in range(len(pivot_pers.index)):
    for j in range(len(pivot_pers.columns)):
        val = pivot_pers.values[i, j]
        if not np.isnan(val):
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color='black' if 0.3 < val < 0.8 else ('white' if val <= 0.3 else 'black'))
style_ax(ax4, 'Persistence Rate\n(NMDA × GABA Heatmap)')

# ── Panel 5: Distractor survival heatmap ─────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
pivot_surv = df.groupby(['nmda_scale', 'gaba_scale'])['survives_distractor'].mean().unstack()
im5 = ax5.imshow(pivot_surv.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, origin='lower')
cb5 = plt.colorbar(im5, ax=ax5, label='Survival rate')
cb5.ax.yaxis.label.set_color(TXT); cb5.ax.tick_params(colors=TXT)
ax5.set_xticks(range(len(pivot_surv.columns)))
ax5.set_xticklabels([f'GABA={g:.1f}' for g in pivot_surv.columns], rotation=30, fontsize=8, color=TICK_C)
ax5.set_yticks(range(len(pivot_surv.index)))
ax5.set_yticklabels([f'NMDA={n:.1f}' for n in pivot_surv.index], fontsize=8, color=TICK_C)
for i in range(len(pivot_surv.index)):
    for j in range(len(pivot_surv.columns)):
        val = pivot_surv.values[i, j]
        if not np.isnan(val):
            ax5.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color='black' if 0.3 < val < 0.8 else ('white' if val <= 0.3 else 'black'))
style_ax(ax5, 'Distractor Survival Rate\n(NMDA × GABA Heatmap)')

# ── Panel 6: Firing-rate variability bars ─────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
std_means = [df[df['condition'] == c]['r1_delay_std'].mean() for c in cond_order]
std_sems  = [df[df['condition'] == c]['r1_delay_std'].sem()  for c in cond_order]
ax6.bar(range(len(cond_order)), std_means,
        color=[color_map[c] for c in cond_order], alpha=0.75,
        yerr=std_sems, capsize=4,
        error_kw=dict(color='black', linewidth=1.5))
ax6.set_xticks(range(len(cond_order)))
ax6.set_xticklabels([c.split('(')[0].strip() for c in cond_order],
                    rotation=40, ha='right', fontsize=7, color=TICK_C)
style_ax(ax6, 'Firing-Rate Variability (σ)\n(Noise / Instability Proxy)', ylabel='Std dev (Hz)')

# ── Panel 7: Delay gain vs SNR scatter ───────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
for cond in conditions:
    sub = df[df['condition'] == cond]
    ax7.scatter(sub['delay_gain'], sub['snr_delay'],
                color=color_map[cond], alpha=0.4, s=12,
                label=cond.split('(')[0].strip())
ax7.set_xlabel('Delay Gain', color=TXT, fontsize=9)
ax7.set_ylabel('SNR Delay', color=TXT, fontsize=9)
ax7.legend(fontsize=6.5, facecolor=LEG_BG, labelcolor=TXT,
           edgecolor=LEG_EDG, loc='upper left')
r, p = stats.spearmanr(df['delay_gain'].dropna(), df['snr_delay'].dropna())
ax7.text(0.98, 0.05, f'ρ={r:.3f}, p={p:.2e}', transform=ax7.transAxes,
         ha='right', va='bottom', fontsize=8, color=ANNOT_C,
         bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#aaaaaa', alpha=0.8))
style_ax(ax7, 'Delay Gain vs SNR\n(Spearman correlation)')

# ── Panel 8: Firing rate → SNR regression ────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
for cond in conditions:
    sub = df[df['condition'] == cond].dropna(subset=['r1_delay_mean', 'snr_delay'])
    ax8.scatter(sub['r1_delay_mean'], sub['snr_delay'],
                color=color_map[cond], alpha=0.3, s=10)
x_clean = df[['r1_delay_mean', 'snr_delay']].dropna()
slope, intercept, r_val, p_val, _ = stats.linregress(x_clean['r1_delay_mean'], x_clean['snr_delay'])
xr = np.linspace(x_clean['r1_delay_mean'].min(), x_clean['r1_delay_mean'].max(), 100)
ax8.plot(xr, slope * xr + intercept, color='black', linewidth=1.8, linestyle='--')
ax8.text(0.98, 0.05, f'r²={r_val**2:.3f}, p={p_val:.2e}',
         transform=ax8.transAxes, ha='right', va='bottom', fontsize=8, color=ANNOT_C,
         bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#aaaaaa', alpha=0.8))
style_ax(ax8, 'Firing Rate → SNR\n(Linear regression)',
         xlabel='Mean delay firing rate (Hz)', ylabel='SNR')

# ── Panel 9: Persistent + survives_distractor by Jp ──────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
jp_vals = sorted(df['Jp'].unique())
x = np.arange(len(jp_vals))
w = 0.35
pers_means = [df[df['Jp'] == jp]['persistent'].astype(float).mean() for jp in jp_vals]
surv_means = [df[df['Jp'] == jp]['survives_distractor'].astype(float).mean() for jp in jp_vals]
b1 = ax9.bar(x - w/2, pers_means, w, label='Persistent', color='#1F77B4', alpha=0.8)
b2 = ax9.bar(x + w/2, surv_means, w, label='Surv. distractor', color='#FF7F0E', alpha=0.8)
ax9.set_xticks(x)
ax9.set_xticklabels([f'Jp={j}' for j in jp_vals], color=TICK_C, fontsize=9)
ax9.set_ylim(0, 1.15)
ax9.legend(fontsize=8, facecolor=LEG_BG, labelcolor=TXT, edgecolor=LEG_EDG)
for b in [b1, b2]:
    for bar in b.patches:
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom',
                 fontsize=8, color=TXT)
_, p_jp = mannwhitneyu(
    df[df['Jp'] == jp_vals[0]]['snr_delay'].dropna(),
    df[df['Jp'] == jp_vals[1]]['snr_delay'].dropna())
sig_jp = '***' if p_jp < 0.001 else ('**' if p_jp < 0.01 else ('*' if p_jp < 0.05 else 'ns'))
ax9.text(0.98, 0.97, f'SNR MW p={p_jp:.3f} {sig_jp}',
         transform=ax9.transAxes, ha='right', va='top', fontsize=8, color=ANNOT_C,
         bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#aaaaaa', alpha=0.8))
style_ax(ax9, 'Persistence & Distractor Survival\nby Jp (synaptic strength)', ylabel='Proportion')

# ── Panel 10: Effect-size bars ────────────────────────────────────────────────
ax10 = fig.add_subplot(gs[3, 0:2])
metrics_plot = list(kw_results.keys())
eta2_vals = [kw_results[m]['eta2'] for m in metrics_plot]
ps_vals   = [kw_results[m]['p']    for m in metrics_plot]
bar_colors = ['#D62728' if p < 0.001 else ('#FF7F0E' if p < 0.01
              else ('#2CA02C' if p < 0.05 else '#9E9E9E')) for p in ps_vals]
bars10 = ax10.barh(metrics_plot, eta2_vals, color=bar_colors, alpha=0.8, height=0.6)
for xv, lbl in [(0.01, 'small'), (0.06, 'medium'), (0.14, 'large')]:
    ax10.axvline(xv, color='#888888', linewidth=0.8, linestyle='--', alpha=0.6)
    ax10.text(xv + 0.001, -0.7, lbl, color='#555555', fontsize=7, va='top')
for bar, eta, p in zip(bars10.patches, eta2_vals, ps_vals):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax10.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
              f'η²={eta:.3f} {sig}', va='center', fontsize=8, color=TXT)
ax10.set_xlabel('Effect size (η²)', color=TXT, fontsize=9)
ax10.tick_params(axis='y', colors=TICK_C, labelsize=8)
style_ax(ax10, 'Effect Sizes – Kruskal-Wallis H-test\n(red=p<0.001, orange=p<0.01, green=p<0.05)')

# ── Panel 11: Post-hoc SNR p-value matrix ────────────────────────────────────
ax11 = fig.add_subplot(gs[3, 2])
ph_snr = posthoc_mw(df, 'snr_delay', conditions)
n = len(conditions)
pmat = np.ones((n, n))
if not ph_snr.empty:
    for _, row in ph_snr.iterrows():
        i = conditions.index(row['G1'])
        j = conditions.index(row['G2'])
        pmat[i, j] = row['p_bonf']; pmat[j, i] = row['p_bonf']
np.fill_diagonal(pmat, np.nan)
log_pmat = np.where(~np.isnan(pmat), -np.log10(np.clip(pmat, 1e-10, 1)), np.nan)
im11 = ax11.imshow(log_pmat, cmap='YlOrRd', vmin=0, vmax=4, aspect='auto')
cb11 = plt.colorbar(im11, ax=ax11, label='-log₁₀(p_bonf)')
cb11.ax.yaxis.label.set_color(TXT); cb11.ax.tick_params(colors=TXT)
short = [c.split('(')[0].strip() for c in conditions]
ax11.set_xticks(range(n)); ax11.set_xticklabels(short, rotation=45, ha='right', fontsize=6.5, color=TICK_C)
ax11.set_yticks(range(n)); ax11.set_yticklabels(short, fontsize=6.5, color=TICK_C)
for i in range(n):
    for j in range(n):
        if not np.isnan(pmat[i, j]):
            sig = '***' if pmat[i,j]<0.001 else ('**' if pmat[i,j]<0.01 else ('*' if pmat[i,j]<0.05 else ''))
            if sig:
                ax11.text(j, i, sig, ha='center', va='center', fontsize=7, color='black')
style_ax(ax11, 'Post-hoc SNR\n(Bonferroni p-value matrix)')

# ── Global title ──────────────────────────────────────────────────────────────
fig.suptitle('Statistical Analysis — Attractor Network Simulations\n'
             'Schizophrenia / OCD Computational Model  |  Loh, Rolls & Deco (2007) Framework',
             color=TXT, fontsize=14, fontweight='bold', y=0.97)

plt.savefig(OUT_PLOT, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nFigure saved → {OUT_PLOT}")
print(f"Report saved → {OUT_REPORT}")
