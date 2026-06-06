"""
Statistical Analysis — lambda_delay
Schizophrenia / OCD Attractor Network Model
Loh, Rolls & Deco (2007) framework

lambda_delay = decay rate of S1 firing during the delay period
  - Low  → stable persistent activity (healthy working memory)
  - High → rapid decay / attractor collapse (cognitive/positive symptoms)

Pipeline:
  1. Full-data overview (both regimes)
  2. Persistent regime: seed-level mean → mean ± SD per condition
  3. Kruskal-Wallis omnibus + Bonferroni post-hoc
  4. Jp effect
  5. Correlations with SNR, robustness, firing rate
  6. Publication-quality 6-panel white-background figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, shapiro, spearmanr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Theme ──────────────────────────────────────────────────────────────────────
BG      = 'white'
PANEL   = 'white'
TXT     = 'black'
GRID_C  = '#dddddd'
SPINE_C = '#aaaaaa'
ANNOT_C = '#222222'
LEG_BG  = 'white'
LEG_EDG = '#aaaaaa'

STAT_BOX = dict(boxstyle='round,pad=0.25', fc='#f0f0f0', ec='#aaaaaa', alpha=0.9)

# ── Condition metadata ────────────────────────────────────────────────────────
COND_META = {
    (0.8, 0.4): dict(label='(0.8/0.4)',       short='(0.8/0.4)',
                     symptom='Cognitive/Negative', color='#D62728', marker='v'),
    (0.9, 0.3): dict(label='(0.9/0.3)', short='(0.9/0.3)',
                     symptom='Positive Sx',        color='#FF7F0E', marker='D'),
    (1.0, 0.3): dict(label='(1.0/0.3)',        short='(1.0/0.3)',
                     symptom='Intermediate',       color='#7F7F7F', marker='s'),
    (1.0, 0.6): dict(label='(1.0/0.6)',        short='(1.0/0.6)',
                     symptom='Normal',             color='#1F77B4', marker='o'),
    (1.1, 0.4): dict(label='(1.1/0.4)',        short='(1.1/0.4)',
                     symptom='Enhanced',           color='#2CA02C', marker='^'),
    (1.1, 0.6): dict(label='(1.1/0.6)', short='(1.1/0.6)',
                     symptom='Max stability',      color='#9467BD', marker='*'),
}
COND_KEYS  = list(COND_META.keys())
JP_VALS    = [1.75, 1.88]
JP_ALPHA   = {1.75: 0.55, 1.88: 1.0}
JP_LW      = {1.75: 0.8,  1.88: 1.8}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('points_results_2.csv')

def cond_label(row):
    k = (row['nmda_scale'], row['gaba_scale'])
    return COND_META[k]['short'] if k in COND_META else f"NMDA={row['nmda_scale']}"

df['cond_short'] = df.apply(cond_label, axis=1)
df['cond_key']   = list(zip(df['nmda_scale'], df['gaba_scale']))

# ── Seed-level aggregation (persistent regime) ────────────────────────────────
pers = df[df['regime'] == 'persistent'].copy()

seed_level = (
    pers.groupby(['Jp', 'nmda_scale', 'gaba_scale', 'seed'])['lambda_delay']
    .mean()
    .reset_index()
)

cond_agg = (
    seed_level.groupby(['Jp', 'nmda_scale', 'gaba_scale'])['lambda_delay']
    .agg(['mean', 'std', 'sem', 'count'])
    .reset_index()
)
cond_agg.columns = ['Jp', 'nmda_scale', 'gaba_scale', 'mean', 'sd', 'sem', 'n']

# ── Helper: Bonferroni post-hoc ───────────────────────────────────────────────
def posthoc_mw(data, col, group_col, groups):
    pairs   = list(combinations(groups, 2))
    results = []
    for g1, g2 in pairs:
        a = data[data[group_col] == g1][col].dropna()
        b = data[data[group_col] == g2][col].dropna()
        if len(a) < 3 or len(b) < 3: continue
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        r = 1 - 2*U / (len(a)*len(b))
        results.append({'G1': g1, 'G2': g2, 'U': U, 'p_raw': p, 'r': r})
    if not results: return pd.DataFrame()
    out = pd.DataFrame(results)
    out['p_bonf'] = np.minimum(out['p_raw'] * len(out), 1.0)
    out['sig']    = out['p_bonf'].apply(
        lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns')))
    return out

def sig_str(p):
    return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════════
lines = []
lines.append('=' * 76)
lines.append('STATISTICAL ANALYSIS — lambda_delay')
lines.append('Attractor Network Simulations | Schizophrenia / OCD Model')
lines.append('Loh, Rolls & Deco (2007)')
lines.append('=' * 76)
lines.append("""
VARIABLE DEFINITION
  lambda_delay = exponential decay rate of S1 pool firing during the delay period.
  Interpretation:
    Low  (≈ 0)   → stable persistent firing → healthy working memory
    High (>> 0)  → rapid attractor collapse → cognitive / positive symptoms
    Negative     → slight firing increase during delay (hyper-stable)
""")

# 1. Overview both regimes
lines.append('─' * 76)
lines.append('1. OVERVIEW BY REGIME')
lines.append('─' * 76)
for regime in ['persistent', 'transient']:
    sub = df[df['regime'] == regime]['lambda_delay']
    lines.append(f"\n  Regime: {regime}  (n={len(sub)})")
    lines.append(f"  Mean={sub.mean():.3f}  SD={sub.std():.3f}  "
                 f"Median={sub.median():.3f}  IQR=[{sub.quantile(.25):.3f}, {sub.quantile(.75):.3f}]")
    w, p = shapiro(sub.sample(min(len(sub), 200), random_state=42))
    lines.append(f"  Shapiro-Wilk (n≤200 subsample): W={w:.3f}, p={p:.4e}  "
                 f"({'non-normal' if p<0.05 else 'normal'})")

U_reg, p_reg = mannwhitneyu(
    df[df['regime']=='persistent']['lambda_delay'],
    df[df['regime']=='transient']['lambda_delay'], alternative='two-sided')
lines.append(f"\n  Mann-Whitney persistent vs transient: U={U_reg:.0f}, p={p_reg:.4e} {sig_str(p_reg)}")

# 2. Persistent regime: descriptive by condition
lines.append('\n' + '─' * 76)
lines.append('2. DESCRIPTIVE STATISTICS — PERSISTENT REGIME (seed-level means, n=19)')
lines.append('─' * 76)
lines.append(f"\n  {'Jp':>5}  {'NMDA':>5}  {'GABA':>5}  {'n':>4}  "
             f"{'Mean':>8}  {'SD':>8}  {'Median':>8}  {'Min':>8}  {'Max':>8}")
lines.append('  ' + '-' * 62)

for _, row in cond_agg.sort_values(['Jp','nmda_scale','gaba_scale']).iterrows():
    sub = seed_level[(seed_level['Jp']==row['Jp']) &
                     (seed_level['nmda_scale']==row['nmda_scale']) &
                     (seed_level['gaba_scale']==row['gaba_scale'])]['lambda_delay']
    lines.append(f"  {row['Jp']:>5.2f}  {row['nmda_scale']:>5.2f}  {row['gaba_scale']:>5.2f}  "
                 f"{int(row['n']):>4}  {row['mean']:>8.3f}  {row['sd']:>8.3f}  "
                 f"{sub.median():>8.3f}  {sub.min():>8.3f}  {sub.max():>8.3f}")

# 3. Normality
lines.append('\n' + '─' * 76)
lines.append('3. NORMALITY CHECK (Shapiro-Wilk, seed-level means per condition)')
lines.append('─' * 76)
n_fails = 0
for (jp, nmda, gaba), grp in seed_level.groupby(['Jp','nmda_scale','gaba_scale']):
    col = grp['lambda_delay'].dropna()
    if len(col) < 3: continue
    w, p = shapiro(col)
    flag = '  ← non-normal' if p < 0.05 else ''
    if p < 0.05: n_fails += 1
    lines.append(f"  Jp={jp} NMDA={nmda} GABA={gaba}  W={w:.3f}  p={p:.4f}{flag}")
lines.append(f"\n  → {n_fails} non-normal distributions. Using non-parametric tests.")

# 4. Kruskal-Wallis per Jp
lines.append('\n' + '─' * 76)
lines.append('4. KRUSKAL-WALLIS H-TEST (across conditions, within each Jp)')
lines.append('─' * 76)

for jp in JP_VALS:
    sub = seed_level[seed_level['Jp'] == jp]
    groups = [sub[sub['nmda_scale']==n][sub['gaba_scale']==g]['lambda_delay'].dropna().values
              for n, g in COND_KEYS
              if len(sub[(sub['nmda_scale']==n)&(sub['gaba_scale']==g)]) >= 3]
    H, p = kruskal(*groups)
    n_tot = sum(len(g) for g in groups)
    eta2  = (H - len(groups) + 1) / (n_tot - len(groups))
    lines.append(f"\n  Jp={jp}: H({len(groups)-1})={H:.3f}, p={p:.4e} {sig_str(p)}, η²={eta2:.3f}")

# Combined
all_groups = [seed_level[(seed_level['nmda_scale']==n)&(seed_level['gaba_scale']==g)]['lambda_delay'].dropna().values
              for n, g in COND_KEYS]
H_all, p_all = kruskal(*all_groups)
n_tot_all = sum(len(g) for g in all_groups)
eta2_all  = (H_all - len(all_groups) + 1) / (n_tot_all - len(all_groups))
lines.append(f"\n  Combined (both Jp): H({len(all_groups)-1})={H_all:.3f}, "
             f"p={p_all:.4e} {sig_str(p_all)}, η²={eta2_all:.3f}")

# 5. Post-hoc pairwise
lines.append('\n' + '─' * 76)
lines.append('5. POST-HOC PAIRWISE MANN-WHITNEY + BONFERRONI (seed-level, both Jp)')
lines.append('─' * 76)

seed_level['cond_label'] = seed_level.apply(
    lambda r: COND_META[(r['nmda_scale'], r['gaba_scale'])]['short'], axis=1)
cond_labels = [COND_META[k]['short'] for k in COND_KEYS]
ph = posthoc_mw(seed_level, 'lambda_delay', 'cond_label', cond_labels)

if not ph.empty:
    lines.append(f"\n  {'Pair':<35} {'U':>10} {'p_bonf':>10}  Sig    r")
    lines.append('  ' + '-' * 65)
    for _, row in ph.sort_values('p_bonf').iterrows():
        lines.append(f"  {row['G1'][:16]:<16} vs {row['G2'][:16]:<16} "
                     f"{row['U']:>10.1f} {row['p_bonf']:>10.4f}  {row['sig']:<5}  {row['r']:>6.3f}")

# 6. Jp effect
lines.append('\n' + '─' * 76)
lines.append('6. Jp EFFECT ON lambda_delay (Mann-Whitney, within each condition)')
lines.append('─' * 76)
lines.append(f"\n  {'Condition':<20} {'U':>8} {'p':>10}  Sig    r")
lines.append('  ' + '-' * 52)
for n, g in COND_KEYS:
    a = seed_level[(seed_level['Jp']==1.75)&(seed_level['nmda_scale']==n)&(seed_level['gaba_scale']==g)]['lambda_delay']
    b = seed_level[(seed_level['Jp']==1.88)&(seed_level['nmda_scale']==n)&(seed_level['gaba_scale']==g)]['lambda_delay']
    if len(a)<3 or len(b)<3: continue
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    r = 1 - 2*U/(len(a)*len(b))
    lines.append(f"  {COND_META[(n,g)]['short']:<20} {U:>8.1f} {p:>10.4f}  {sig_str(p):<5}  {r:>6.3f}")

# 7. Correlations
lines.append('\n' + '─' * 76)
lines.append('7. SPEARMAN CORRELATIONS WITH lambda_delay (persistent regime, all rows)')
lines.append('─' * 76)
corr_metrics = {
    'snr_delay':           'SNR during delay',
    'attractor_robustness':'Attractor robustness',
    'r1_delay_mean':       'Mean S1 firing rate (delay)',
    'r1_delay_std':        'Firing-rate variability σ',
    'delay_gain':          'Delay gain',
    'r1_delay_min':        'Min S1 firing rate (delay)',
    'collapse_time':       'Collapse time',
}
lines.append(f"\n  {'Metric':<35} {'ρ':>7} {'p':>12}  Sig")
lines.append('  ' + '-' * 58)
for col, label in corr_metrics.items():
    if col not in pers.columns: continue
    clean = pers[['lambda_delay', col]].dropna()
    rho, p = spearmanr(clean['lambda_delay'], clean[col])
    lines.append(f"  {label:<35} {rho:>7.3f} {p:>12.4e}  {sig_str(p)}")

lines.append('\n' + '=' * 76)
lines.append('END OF REPORT')
lines.append('=' * 76)

report_text = '\n'.join(lines)
print(report_text)

OUT_REPORT = 'lambda_delay_report.txt'
with open(OUT_REPORT, 'w', encoding='utf-8') as f:
    f.write(report_text)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE  —  6 panels, white background
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                       left=0.07, right=0.97, top=0.91, bottom=0.10)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TXT, fontsize=11, fontweight='bold', pad=9)
    ax.set_xlabel(xlabel, color=TXT, fontsize=9)
    ax.set_ylabel(ylabel, color=TXT, fontsize=9)
    ax.tick_params(colors=TXT, labelsize=8.5)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.grid(axis='y', color=GRID_C, linewidth=0.6, linestyle='--', zorder=0)

x_pos = np.arange(len(COND_KEYS))

# ── Panel A: Mean ± SD per condition × Jp (main reviewer plot) ────────────────
ax_a = fig.add_subplot(gs[0, 0:2])   # wider panel — the key result

for jp_idx, jp in enumerate(JP_VALS):
    sub = cond_agg[cond_agg['Jp'] == jp]
    jitter = (jp_idx - 0.5) * 0.18

    xs, means, sds, cols, mks = [], [], [], [], []
    for xi, (nmda, gaba) in enumerate(COND_KEYS):
        row = sub[(np.isclose(sub['nmda_scale'], nmda)) & (np.isclose(sub['gaba_scale'], gaba))]
        means.append(float(row['mean'].iloc[0]) if not row.empty else np.nan)
        sds.append(float(row['sd'].iloc[0])     if not row.empty else np.nan)
        xs.append(xi + jitter)
        cols.append(COND_META[(nmda, gaba)]['color'])
        mks.append(COND_META[(nmda, gaba)]['marker'])

    means = np.array(means); sds = np.array(sds)
    valid = ~np.isnan(means)

    # connecting line
    ax_a.plot(np.array(xs)[valid], means[valid],
              color='#555555', lw=0.9, ls='--', alpha=JP_ALPHA[jp]*0.6, zorder=2)

    # error bars + markers
    for x_i, m, sd, col, mk in zip(xs, means, sds, cols, mks):
        if np.isnan(m): continue
        ax_a.errorbar(x_i, m, yerr=sd, fmt='none',
                      ecolor=col, elinewidth=2.2, capsize=7, capthick=2.2,
                      alpha=JP_ALPHA[jp], zorder=3)
        ms = 12 if mk == '*' else 10
        ax_a.plot(x_i, m, marker=mk, color=col, markersize=ms,
                  markeredgecolor='black', markeredgewidth=JP_LW[jp],
                  alpha=JP_ALPHA[jp], zorder=4, linestyle='none')

# KW annotation
H_a, p_a = kruskal(*[seed_level[(seed_level['nmda_scale']==n)&(seed_level['gaba_scale']==g)]['lambda_delay'].dropna()
                      for n, g in COND_KEYS])
n_tot_a = sum(len(seed_level[(seed_level['nmda_scale']==n)&(seed_level['gaba_scale']==g)]['lambda_delay'].dropna())
              for n, g in COND_KEYS)
eta2_a = (H_a - len(COND_KEYS) + 1) / (n_tot_a - len(COND_KEYS))
ax_a.text(0.99, 0.97,
          f'Kruskal-Wallis: H({len(COND_KEYS)-1})={H_a:.1f}, p={p_a:.2e} {sig_str(p_a)}\nη²={eta2_a:.3f}',
          transform=ax_a.transAxes, ha='right', va='top', fontsize=8.5,
          color=ANNOT_C, bbox=STAT_BOX)

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels([COND_META[k]['label'] for k in COND_KEYS], fontsize=9, color=TXT)
ax_a.set_ylabel('λ_delay (decay rate)', color=TXT, fontsize=10)
ax_a.axhline(0, color='#CC0000', lw=1.2, ls=':', alpha=0.6, label='λ=0 (no decay)')

# Jp legend
jp_handles = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor='#555555',
           markeredgecolor='black', markersize=9, markeredgewidth=JP_LW[jp],
           alpha=JP_ALPHA[jp], label=f'Jp = {jp}')
    for jp in JP_VALS
]
jp_handles.append(Line2D([0],[0], color='#CC0000', lw=1.2, ls=':', label='λ = 0'))
ax_a.legend(handles=jp_handles, fontsize=8.5, facecolor=LEG_BG,
            edgecolor=LEG_EDG, labelcolor=TXT, loc='upper left')

# symptom bands
for xi, (nmda, gaba) in enumerate(COND_KEYS):
    ax_a.axvspan(xi-0.5, xi+0.5, color=COND_META[(nmda,gaba)]['color'], alpha=0.06, zorder=0)

style_ax(ax_a, 'λ_delay per Condition — Mean ± SD across Seeds (n=19 per point)\n'
               'Higher λ = faster decay = less stable attractor',
         ylabel='λ_delay (decay rate)')

# ── Panel B: Violin / strip — full persistent data ────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])

vdata  = [pers[(pers['nmda_scale']==n)&(pers['gaba_scale']==g)]['lambda_delay'].dropna().values
          for n, g in COND_KEYS]
parts  = ax_b.violinplot(vdata, positions=range(len(COND_KEYS)),
                         showmedians=True, showextrema=False, widths=0.65)
for pc, (n, g) in zip(parts['bodies'], COND_KEYS):
    pc.set_facecolor(COND_META[(n,g)]['color']); pc.set_alpha(0.45)
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
for xi, (vals, (n, g)) in enumerate(zip(vdata, COND_KEYS)):
    jit = np.random.default_rng(42).normal(0, 0.07, len(vals))
    ax_b.scatter(xi + jit, vals, s=6, alpha=0.3, color=COND_META[(n,g)]['color'], zorder=3)
ax_b.axhline(0, color='#CC0000', lw=1.1, ls=':', alpha=0.6)
ax_b.set_xticks(range(len(COND_KEYS)))
ax_b.set_xticklabels([COND_META[k]['label'] for k in COND_KEYS],
                     rotation=30, ha='right', fontsize=8, color=TXT)
ax_b.set_ylabel('λ_delay', color=TXT, fontsize=9)
style_ax(ax_b, 'Distribution by Condition\n(persistent regime, all rows)')

# ── Panel C: lambda_delay ~ SNR scatter + regression ─────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

for n, g in COND_KEYS:
    sub = pers[(pers['nmda_scale']==n)&(pers['gaba_scale']==g)]
    ax_c.scatter(sub['lambda_delay'], sub['snr_delay'],
                 color=COND_META[(n,g)]['color'], alpha=0.35, s=10,
                 label=COND_META[(n,g)]['label'])
clean = pers[['lambda_delay','snr_delay']].dropna()
slope, intercept, r_val, p_val, _ = stats.linregress(clean['lambda_delay'], clean['snr_delay'])
xr = np.linspace(clean['lambda_delay'].min(), clean['lambda_delay'].max(), 200)
ax_c.plot(xr, slope*xr+intercept, color='black', lw=1.8, ls='--', zorder=5)
rho, p_rho = spearmanr(clean['lambda_delay'], clean['snr_delay'])
ax_c.text(0.97, 0.97,
          f'ρ = {rho:.3f}  {sig_str(p_rho)}\nr² = {r_val**2:.3f}',
          transform=ax_c.transAxes, ha='right', va='top', fontsize=8.5,
          color=ANNOT_C, bbox=STAT_BOX)
ax_c.legend(fontsize=7, facecolor=LEG_BG, edgecolor=LEG_EDG, labelcolor=TXT,
            loc='upper right', ncol=2)
style_ax(ax_c, 'λ_delay vs SNR\n(Spearman + OLS)',
         xlabel='λ_delay (decay rate)', ylabel='SNR – delay period')

# ── Panel D: lambda_delay ~ attractor_robustness ──────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

for n, g in COND_KEYS:
    sub = pers[(pers['nmda_scale']==n)&(pers['gaba_scale']==g)]
    ax_d.scatter(sub['lambda_delay'], sub['attractor_robustness'],
                 color=COND_META[(n,g)]['color'], alpha=0.35, s=10)
clean_d = pers[['lambda_delay','attractor_robustness']].dropna()
slope_d, int_d, r_d, p_d, _ = stats.linregress(clean_d['lambda_delay'], clean_d['attractor_robustness'])
xr_d = np.linspace(clean_d['lambda_delay'].min(), clean_d['lambda_delay'].max(), 200)
ax_d.plot(xr_d, slope_d*xr_d+int_d, color='black', lw=1.8, ls='--', zorder=5)
rho_d, p_rho_d = spearmanr(clean_d['lambda_delay'], clean_d['attractor_robustness'])
ax_d.axhline(0, color='#CC0000', lw=1, ls=':', alpha=0.5)
ax_d.text(0.97, 0.97,
          f'ρ = {rho_d:.3f}  {sig_str(p_rho_d)}\nr² = {r_d**2:.3f}',
          transform=ax_d.transAxes, ha='right', va='top', fontsize=8.5,
          color=ANNOT_C, bbox=STAT_BOX)
style_ax(ax_d, 'λ_delay vs Attractor Robustness\n(Spearman + OLS)',
         xlabel='λ_delay (decay rate)', ylabel='Attractor robustness')

# ── Panel E: Jp comparison boxplot ───────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])

positions_jp = []
tick_labels_e = []
for xi, (nmda, gaba) in enumerate(COND_KEYS):
    for jp_idx, jp in enumerate(JP_VALS):
        xp = xi * 2.8 + jp_idx * 1.1
        vals = seed_level[(seed_level['Jp']==jp) &
                          (seed_level['nmda_scale']==nmda) &
                          (seed_level['gaba_scale']==gaba)]['lambda_delay'].dropna()
        bp = ax_e.boxplot(vals, positions=[xp], widths=0.8,
                          patch_artist=True, notch=False,
                          medianprops=dict(color='black', lw=2),
                          whiskerprops=dict(color='#555555'),
                          capprops=dict(color='#555555'),
                          flierprops=dict(marker='.', color=COND_META[(nmda,gaba)]['color'],
                                         alpha=0.4, markersize=4))
        alpha = JP_ALPHA[jp]
        bp['boxes'][0].set_facecolor(COND_META[(nmda,gaba)]['color'])
        bp['boxes'][0].set_alpha(alpha)
        if jp_idx == 0:
            positions_jp.append(xi*2.8 + 0.55)
            tick_labels_e.append(COND_META[(nmda,gaba)]['label'])

ax_e.set_xticks(positions_jp)
ax_e.set_xticklabels(tick_labels_e, rotation=28, ha='right', fontsize=7.5, color=TXT)
ax_e.axhline(0, color='#CC0000', lw=1, ls=':', alpha=0.5)
jp_patches = [Patch(facecolor='#555555', alpha=a, edgecolor='black',
                    label=f'Jp={jp}') for jp, a in JP_ALPHA.items()]
ax_e.legend(handles=jp_patches, fontsize=8, facecolor=LEG_BG,
            edgecolor=LEG_EDG, labelcolor=TXT)
ax_e.set_ylabel('λ_delay', color=TXT, fontsize=9)
style_ax(ax_e, 'λ_delay by Condition × Jp\n(seed-level means, boxplot)')

# ── Shared condition legend at bottom ─────────────────────────────────────────
cond_handles = [
    Line2D([0],[0], marker=COND_META[k]['marker'], color='none',
           markerfacecolor=COND_META[k]['color'], markeredgecolor='black',
           markersize=9 if COND_META[k]['marker']!='*' else 12,
           label=f"{COND_META[k]['label']} — {COND_META[k]['symptom']}")
    for k in COND_KEYS
]
fig.legend(handles=cond_handles, title='Condition (NMDA/GABA) — Symptom category',
           title_fontsize=9, fontsize=8.5, ncol=3,
           loc='lower center', bbox_to_anchor=(0.5, 0.01),
           facecolor=LEG_BG, edgecolor=LEG_EDG, labelcolor=TXT,
           framealpha=1.0).get_title().set_color(TXT)

fig.suptitle(
    'Statistical Analysis — λ_delay (Attractor Decay Rate)\n'
    'Schizophrenia / OCD Computational Model  |  Loh, Rolls & Deco (2007)',
    color=TXT, fontsize=13, fontweight='bold', y=0.975)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

OUT_FIG = 'lambda_delay_analysis.png'
plt.savefig(OUT_FIG, dpi=180, bbox_inches='tight', facecolor=BG)
print(f'\nFigure → {OUT_FIG}')
print(f'Report → {OUT_REPORT}')
