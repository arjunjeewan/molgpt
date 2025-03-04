"""
Aggregate datasets/guaca_metrics_*.json into the comprehensive ablation tables.

  Table A (component ablation, unconditional): one row per config with RoPE/SwiGLU/RMSNorm
           shown as checked (#) / crossed (.) plus distribution-learning metrics
           (mean +/- std over seeds).
  Table B (generation modes, baseline vs modern): per conditioning mode, baseline vs modern
           with validity/unique/novelty + MAD (property control) + scaffold-match.

Writes experiments/guaca_ablation_results.md and datasets/guaca_results_long.csv.
"""
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GLOB = os.path.join(ROOT, 'datasets', 'guaca_metrics_*.json')

CHK, CRS = '✓', '✗'   # check / cross
CONFIG_ORDER = ['baseline', 'rope', 'swiglu', 'rmsnorm', 'modern']
CONFIG_LABEL = {'baseline': 'baseline', 'rope': '+RoPE', 'swiglu': '+SwiGLU',
                'rmsnorm': '+RMSNorm', 'modern': 'modern (all)'}


def load():
    recs = []
    for p in glob.glob(GLOB):
        with open(p) as f:
            recs.append(json.load(f))
    return recs


def ms(vals):
    """mean+/-std string; std shown only when >1 seed."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return '-'
    m = np.mean(vals)
    if len(vals) == 1:
        return f'{m:.3f}'
    return f'{m:.3f}±{np.std(vals, ddof=0):.3f}'


def mode_label(r):
    if r['mode'] == 'unconditional':
        return 'unconditional'
    if r['mode'] == 'conditional':
        return f"prop:{r['props'][0]}" if len(r['props']) == 1 else f"prop:{'+'.join(r['props'])}"
    if r['mode'] == 'scaffold':
        return 'scaffold'
    return f"scaffold+{'+'.join(r['props'])}"


def table_A(recs):
    rows = [r for r in recs if r['mode'] == 'unconditional']
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r['config'], []).append(r)
    out = []
    out.append('### Table A — Component ablation (unconditional GuacaMol, T=1.0)')
    out.append('')
    out.append(f'Each component {CHK}=on / {CRS}=off. Metrics are mean over seeds '
               '(±std where >1 seed). Validity/Unique/Novelty and KL-div: higher=better. '
               'FCD: lower=better (distance to held-out reference).')
    out.append('')
    out.append('| Config | RoPE | SwiGLU | RMSNorm | Validity | Unique | Novelty | KL-div | FCD |')
    out.append('|---|:--:|:--:|:--:|--:|--:|--:|--:|--:|')
    for cfg in CONFIG_ORDER:
        rs = by_cfg.get(cfg, [])
        if not rs:
            continue
        e = rs[0]
        r_ = CHK if e['use_rope'] else CRS
        s_ = CHK if e['use_swiglu'] else CRS
        n_ = CHK if e['use_rmsnorm'] else CRS
        seeds = sorted(x['seed'] for x in rs)
        out.append('| {} ({}s: {}) | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            CONFIG_LABEL[cfg], len(rs), ','.join(map(str, seeds)), r_, s_, n_,
            ms([x['validity'] for x in rs]), ms([x['unique'] for x in rs]),
            ms([x['novelty'] for x in rs]), ms([x.get('kl_score') for x in rs]),
            ms([x.get('fcd') for x in rs])))
    return '\n'.join(out)


def table_B(recs):
    rows = [r for r in recs if r['mode'] != 'unconditional']
    # group by (mode_label, config)
    groups = {}
    for r in rows:
        groups.setdefault((mode_label(r), r['config']), []).append(r)
    labels = sorted({k[0] for k in groups})
    out = []
    out.append('### Table B — Generation modes, baseline vs modern (GuacaMol, T=1.0)')
    out.append('')
    out.append('Validity/Unique/Novelty: higher=better. MAD (mean abs deviation of the generated '
               'property from the target, averaged over the target grid): lower=better. '
               'Scaffold-match: fraction of generated mols whose Murcko scaffold equals the '
               'conditioning scaffold (higher=better).')
    out.append('')
    out.append('| Mode | Arch | Validity | Unique | Novelty | MAD | Scaffold-match |')
    out.append('|---|---|--:|--:|--:|--:|--:|')
    for lab in labels:
        for cfg in ['baseline', 'modern']:
            rs = groups.get((lab, cfg))
            if not rs:
                continue
            # MAD: average across the props present
            mad_keys = sorted({k for r in rs for k in r if k.startswith('mad_')})
            if mad_keys:
                mad_str = '; '.join(f"{k[4:]}={ms([r.get(k) for r in rs])}" for k in mad_keys)
            else:
                mad_str = '-'
            sm = ms([r.get('scaffold_match') for r in rs]) if any('scaffold_match' in r for r in rs) else '-'
            out.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                lab, cfg, ms([r['validity'] for r in rs]), ms([r['unique'] for r in rs]),
                ms([r['novelty'] for r in rs]), mad_str, sm))
    return '\n'.join(out)


def long_csv(recs):
    import csv
    cols = ['run_name', 'mode', 'config', 'use_rope', 'use_swiglu', 'use_rmsnorm', 'seed',
            'temp', 'validity', 'unique', 'novelty', 'kl_score', 'fcd', 'fcd_score',
            'mad_qed', 'mad_sas', 'mad_logp', 'mad_tpsa', 'scaffold_match']
    out = os.path.join(ROOT, 'datasets', 'guaca_results_long.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(recs, key=lambda x: (x['mode'], x['config'], x.get('seed', 0))):
            w.writerow(r)
    return out


def main():
    recs = load()
    if not recs:
        print('no guaca_metrics_*.json found yet'); return
    a = table_A(recs)
    b = table_B(recs)
    md = '# GuacaMol component ablation — comprehensive results\n\n' + a + '\n\n' + b + '\n'
    out = os.path.join(HERE, 'guaca_ablation_results.md')
    with open(out, 'w') as f:
        f.write(md)
    csvp = long_csv(recs)
    print(md)
    print('\nwrote', out)
    print('wrote', csvp)


if __name__ == '__main__':
    main()
