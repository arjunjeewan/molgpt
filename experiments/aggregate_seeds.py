"""
Aggregate the multi-seed ablation into robust metrics.

Reads:
  datasets/sweep_seed_<tag>.csv           per-(arch,seed) pure-temperature frontier
  datasets/moses_metrics_<tag>_T1.0.json  per-(arch,seed) full MOSES metrics

Produces (printed + written to experiments/multiseed_results.md, datasets/multiseed_summary.csv):
  - frontier: mean +/- std across seeds per (arch, temperature)
  - paired modern-vs-baseline Delta-novelty per temperature (per-seed deltas, mean, paired t-test)
  - full-MOSES mean +/- std across seeds per arch at T=1.0
Only tags whose files exist are included, so this can be run on partial results.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'datasets')

# tag -> (arch, seed). e25 is the longer-budget run, handled separately.
TAGS = {
    'base_s42': ('baseline', 42), 'modern_s42': ('modern', 42),
    'base_s1': ('baseline', 1),   'modern_s1': ('modern', 1),
    'base_s2': ('baseline', 2),   'modern_s2': ('modern', 2),
}
FRONTIER_METRICS = ['valid', 'unique', 'novelty', 'intdiv1']


def _fmt(m, s):
    return f'{m:.4f}' if (s is None or np.isnan(s)) else f'{m:.4f}+/-{s:.4f}'


def load_frontier():
    rows = []
    for tag, (arch, seed) in TAGS.items():
        f = os.path.join(DATA, f'sweep_seed_{tag}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['arch'] = arch
        df['seed'] = seed
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def frontier_report(long):
    out = ['## Decoding frontier: mean +/- std across seeds\n']
    g = long.groupby(['arch', 'temp'])
    agg = g[FRONTIER_METRICS].agg(['mean', 'std', 'count'])
    seeds_by = long.groupby(['arch', 'temp'])['seed'].nunique()
    lines = []
    for (arch, temp), _ in agg.iterrows():
        n = int(seeds_by.loc[(arch, temp)])
        cells = []
        for m in FRONTIER_METRICS:
            cells.append(_fmt(agg.loc[(arch, temp), (m, 'mean')], agg.loc[(arch, temp), (m, 'std')]))
        lines.append((arch, temp, n, *cells))
    hdr = '| arch | T | n | validity | unique | novelty | IntDiv1 |'
    sep = '|---|---|---|---|---|---|---|'
    out += [hdr, sep]
    for arch, temp, n, *cells in sorted(lines, key=lambda r: (r[0], r[1])):
        out.append(f'| {arch} | {temp} | {n} | ' + ' | '.join(cells) + ' |')
    out.append('')
    return '\n'.join(out), agg


def paired_novelty(long):
    """Per-temperature paired Delta = novelty(modern) - novelty(baseline), matched by seed."""
    out = ['## Paired novelty edge (modern - baseline), matched by seed\n']
    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        have_scipy = False
    piv = long.pivot_table(index=['temp', 'seed'], columns='arch', values='novelty').reset_index()
    rows = []
    for temp in sorted(piv['temp'].unique()):
        sub = piv[piv['temp'] == temp].dropna(subset=['baseline', 'modern'])
        deltas = (sub['modern'] - sub['baseline']).values
        if len(deltas) == 0:
            continue
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float('nan')
        if have_scipy and len(deltas) > 1 and np.std(deltas) > 0:
            t, p = stats.ttest_rel(sub['modern'].values, sub['baseline'].values)
            tp = f'{t:.3f}' if not np.isnan(t) else 'na'
            pp = f'{p:.4f}' if not np.isnan(p) else 'na'
        else:
            tp, pp = 'na', 'na'
        per_seed = ', '.join(f'{d:+.4f}' for d in deltas)
        rows.append((temp, len(deltas), per_seed, mean_d, std_d, tp, pp))
    out.append('| T | n | per-seed deltas | mean delta | std | t | p (paired) |')
    out.append('|---|---|---|---|---|---|---|')
    for temp, n, per_seed, mean_d, std_d, tp, pp in rows:
        sd = '' if np.isnan(std_d) else f'{std_d:.4f}'
        out.append(f'| {temp} | {n} | {per_seed} | {mean_d:+.4f} | {sd} | {tp} | {pp} |')
    out.append('')
    out.append('_Positive delta = modernized more novel at matched temperature. '
               'With n=3 the paired t-test is low-power; sign-consistency across seeds is the '
               'primary evidence._\n')
    return '\n'.join(out)


def full_moses_report():
    out = ['## Full MOSES metrics at T=1.0: mean +/- std across seeds\n']
    recs = []
    for tag, (arch, seed) in TAGS.items():
        f = os.path.join(DATA, f'moses_metrics_{tag}_T1.0.json')
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            r = json.load(fh)
        r['arch'] = arch
        r['seed'] = seed
        recs.append(r)
    if not recs:
        out.append('_(no full-MOSES json files found yet)_\n')
        return '\n'.join(out), None
    df = pd.DataFrame(recs)
    drop = {'tag', 'temp', 'seed'}
    metric_cols = [c for c in df.columns if c not in drop and c != 'arch'
                   and pd.api.types.is_numeric_dtype(df[c])]
    # focused, ordered subset if present
    pref = ['valid', 'unique@1000', 'unique@10000', 'Novelty', 'IntDiv', 'IntDiv2',
            'FCD/Test', 'SNN/Test', 'Frag/Test', 'Scaf/Test', 'Filters']
    cols = [c for c in pref if c in metric_cols] + [c for c in metric_cols if c not in pref]
    g = df.groupby('arch')[cols].agg(['mean', 'std', 'count'])
    archs = list(g.index)
    out.append('| metric | ' + ' | '.join(archs) + ' |')
    out.append('|---|' + '|'.join(['---'] * len(archs)) + '|')
    for m in cols:
        cells = []
        for a in archs:
            cells.append(_fmt(g.loc[a, (m, 'mean')], g.loc[a, (m, 'std')]))
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    n_by = df.groupby('arch')['seed'].nunique().to_dict()
    out.append('')
    out.append('n seeds per arch: ' + ', '.join(f'{a}={n_by[a]}' for a in archs) + '\n')
    return '\n'.join(out), df


def longer_run_report(long):
    f = os.path.join(DATA, 'sweep_seed_modern_e25.csv')
    if not os.path.exists(f):
        return ''
    e25 = pd.read_csv(f)
    out = ['## Longer modernized run (25 epochs, seed 42) vs 10-epoch modern mean\n']
    base10 = long[(long['arch'] == 'modern')].groupby('temp')[FRONTIER_METRICS].mean()
    out.append('| T | metric | modern 10ep (seed-mean) | modern 25ep |')
    out.append('|---|---|---|---|')
    for _, row in e25.iterrows():
        t = row['temp']
        if t not in base10.index:
            continue
        for m in ['valid', 'novelty']:
            out.append(f'| {t} | {m} | {base10.loc[t, m]:.4f} | {row[m]:.4f} |')
    out.append('')
    return '\n'.join(out)


def main():
    long = load_frontier()
    parts = ['# Multi-seed ablation results\n',
             f'_generated by aggregate_seeds.py; seeds reused 42 + new 1,2_\n']
    if long is None:
        parts.append('_(no sweep_seed_*.csv files found yet)_')
        print('\n'.join(parts))
        return
    long.to_csv(os.path.join(DATA, 'multiseed_frontier_long.csv'), index=False)

    front_md, agg = frontier_report(long)
    parts.append(front_md)
    parts.append(paired_novelty(long))
    fm_md, _ = full_moses_report()
    parts.append(fm_md)
    lr = longer_run_report(long)
    if lr:
        parts.append(lr)

    report = '\n'.join(parts)
    print(report)
    with open(os.path.join(HERE, 'multiseed_results.md'), 'w') as f:
        f.write(report)
    print('\nwrote', os.path.join(HERE, 'multiseed_results.md'))
    print('wrote', os.path.join(DATA, 'multiseed_frontier_long.csv'))


if __name__ == '__main__':
    main()
