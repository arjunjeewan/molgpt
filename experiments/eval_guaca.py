"""
GuacaMol evaluation for the ablation. Rebuilds a model from its sidecar JSON, generates,
and computes mode-appropriate metrics:

  unconditional  : Validity, Uniqueness, Novelty, KL-div score, FCD  (distribution learning)
  conditional    : per-target Validity/Unique/Novelty + MAD (mean abs deviation from target)
  scaffold       : per-scaffold Validity/Unique/Novelty + Murcko-match fraction + scaf Tanimoto
  scaffold+prop  : both of the above

The KL/FCD reference (a fixed held-out sample) is identical for every config, so its
descriptors / internal-similarity / FCD stats are computed once and cached to disk.
Results -> datasets/guaca_metrics_<run_name>_T<temp>.json   (+ raw generations csv)
"""
import os, sys, re, math, json, pickle, argparse
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # molgpt/
sys.path.insert(0, os.path.join(ROOT, 'generate')) # sample, get_mol, sascorer
sys.path.insert(0, HERE)                           # model_ablate
from utils import sample, canonic_smiles, check_novelty
from get_mol import get_mol
from model_ablate import GPT, GPTConfig
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sascorer import calculateScore
RDLogger.DisableLog('rdApp.*')

from guacamol.utils.chemistry import (calculate_pc_descriptors, continuous_kldiv,
                                       discrete_kldiv, calculate_internal_pairwise_similarities)
from fcd_torch import FCD as FCDMetric

WHOLE_STRING = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
REGEX = re.compile(PATTERN)
PC_SUBSET = ['BertzCT', 'MolLogP', 'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors',
             'NumRotatableBonds', 'NumAliphaticRings', 'NumAromaticRings']

# property-target grids (guacamol branch of generate/generate.py)
PROP2VALUE = {
    'qed': [0.3, 0.5, 0.7], 'sas': [2.0, 3.0, 4.0], 'logp': [2.0, 4.0, 6.0], 'tpsa': [40.0, 80.0, 120.0],
    'tpsa_logp': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0], [40.0, 6.0], [80.0, 6.0], [120.0, 6.0]],
    'sas_logp': [[2.0, 2.0], [2.0, 4.0], [2.0, 6.0], [3.0, 2.0], [3.0, 4.0], [3.0, 6.0], [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]],
    'tpsa_sas': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 3.0], [80.0, 3.0], [120.0, 3.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0]],
    'tpsa_logp_sas': [[40.0, 2.0, 2.0], [40.0, 2.0, 4.0], [40.0, 6.0, 4.0], [40.0, 6.0, 2.0], [80.0, 6.0, 4.0], [80.0, 2.0, 4.0], [80.0, 2.0, 2.0], [80.0, 6.0, 2.0]],
}
SCAFFOLDS = ['O=C(Cc1ccccc1)NCc1ccccc1', 'c1cnc2[nH]ccc2c1', 'c1ccc(-c2ccnnc2)cc1',
             'c1ccc(-n2cnc3ccccc32)cc1', 'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1']
PROP_FN = {'qed': lambda m: QED.qed(m), 'sas': lambda m: calculateScore(m),
           'logp': lambda m: Crippen.MolLogP(m), 'tpsa': lambda m: CalcTPSA(m)}


def build_stoi():
    chars = sorted(set(WHOLE_STRING))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, {i: c for c, i in stoi.items()}


def load_model(run_name):
    wdir = os.path.join(ROOT, '..', 'cond_gpt', 'weights')
    with open(os.path.join(wdir, f'{run_name}.json')) as f:
        sc = json.load(f)
    mconf = GPTConfig(sc['vocab_size'], sc['max_len'], num_props=sc['num_props'],
                      n_layer=sc['n_layer'], n_head=sc['n_head'], n_embd=sc['n_embd'],
                      scaffold=sc['scaffold'], scaffold_maxlen=sc['scaffold_max_len'],
                      lstm=False, lstm_layers=0,
                      use_rope=sc['use_rope'], use_swiglu=sc['use_swiglu'], use_rmsnorm=sc['use_rmsnorm'])
    model = GPT(mconf)
    model.load_state_dict(torch.load(os.path.join(wdir, f'{run_name}.pt'), map_location='cpu'))
    model.to('cuda').eval()
    return model, sc


def gen_batch(model, stoi, itos, block_size, temp, batch_size, prop=None, scaffold=None):
    x = torch.tensor([stoi[s] for s in REGEX.findall('C')], dtype=torch.long)[None, ...].repeat(batch_size, 1).to('cuda')
    p = None if prop is None else prop.to('cuda')
    sca = None if scaffold is None else scaffold.to('cuda')
    y = sample(model, x, block_size, temperature=temp, sample=True, top_k=None, prop=p, scaffold=sca)
    out = []
    for row in y:
        s = ''.join(itos[int(i)] for i in row).replace('<', '')
        m = get_mol(s)
        if m:
            out.append(Chem.MolToSmiles(m))
    return out


def basic_metrics(gen_smiles, n_attempted, train_set):
    valid = len(gen_smiles)
    canon = [c for c in (canonic_smiles(s) for s in gen_smiles) if c is not None]
    uniq = set(canon)
    novelty = check_novelty(list(uniq), train_set) / 100.0 if uniq else 0.0
    return {'validity': round(valid / n_attempted, 4),
            'unique': round(len(uniq) / valid, 4) if valid else 0.0,
            'novelty': round(novelty, 4)}, list(uniq)


# ---------------------------------------------------------------------------
# distribution-learning reference (cached: identical for every config)
# ---------------------------------------------------------------------------
def get_reference(ref_smiles, n=10000):
    cache = os.path.join(ROOT, 'datasets', f'guaca_ref_cache_n{n}.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    print('building reference cache (one-time)...', flush=True)
    rng = np.random.RandomState(42)
    ref = list(rng.choice(ref_smiles, size=min(n, len(ref_smiles)), replace=False))
    d_chembl = calculate_pc_descriptors(ref, PC_SUBSET)
    chembl_sim = calculate_internal_pairwise_similarities(ref).max(axis=1)
    fcd = FCDMetric(n_jobs=8, device='cuda', batch_size=512)
    pref = fcd.precalc(ref)
    out = {'ref_smiles': ref, 'd_chembl': d_chembl, 'chembl_sim': chembl_sim, 'pref': pref}
    with open(cache, 'wb') as f:
        pickle.dump(out, f)
    return out


def distribution_metrics(unique_smiles, ref):
    # KL-div score (guacamol KLDivBenchmark formulation)
    d_sampled = calculate_pc_descriptors(unique_smiles, PC_SUBSET)
    kldivs = {}
    for i in range(4):
        kldivs[PC_SUBSET[i]] = continuous_kldiv(X_baseline=ref['d_chembl'][:, i], X_sampled=d_sampled[:, i])
    for i in range(4, 9):
        kldivs[PC_SUBSET[i]] = discrete_kldiv(X_baseline=ref['d_chembl'][:, i], X_sampled=d_sampled[:, i])
    sampled_sim = calculate_internal_pairwise_similarities(unique_smiles).max(axis=1)
    kldivs['internal_similarity'] = continuous_kldiv(X_baseline=ref['chembl_sim'], X_sampled=sampled_sim)
    kl_score = float(np.mean([np.exp(-v) for v in kldivs.values()]))
    # FCD (raw) + guacamol FCD score exp(-0.2*FCD)
    fcd = FCDMetric(n_jobs=8, device='cuda', batch_size=512)
    fcd_raw = float(fcd(gen=unique_smiles, pref=ref['pref']))
    return {'kl_score': round(kl_score, 4), 'fcd': round(fcd_raw, 4),
            'fcd_score': round(float(np.exp(-0.2 * fcd_raw)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--gen_size', type=int, default=10000, help='unconditional total; per-condition uses --cond_size')
    ap.add_argument('--cond_size', type=int, default=2000, help='molecules per property/scaffold condition')
    ap.add_argument('--batch_size', type=int, default=512)
    a = ap.parse_args()

    model, sc = load_model(a.run_name)
    stoi, itos = build_stoi()
    block_size = sc['max_len']
    scaffold_max_len = sc['scaffold_max_len']

    data = pd.read_csv(os.path.join(ROOT, 'datasets', f"{sc['data_name']}.csv"))
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    split_col = 'split' if 'moses' in sc['data_name'] else 'source'
    train_set = set(data[data[split_col] == 'train']['smiles'].map(lambda s: str(s).strip()))
    holdout = data[data[split_col] == ('test' if 'moses' in sc['data_name'] else 'test')]['smiles'].map(lambda s: str(s).strip()).tolist()

    rec = {'run_name': a.run_name, 'config': sc['config'], 'use_rope': sc['use_rope'],
           'use_swiglu': sc['use_swiglu'], 'use_rmsnorm': sc['use_rmsnorm'], 'seed': sc['seed'],
           'num_props': sc['num_props'], 'props': sc['props'], 'scaffold': sc['scaffold'], 'temp': a.temp}

    all_gen = []

    if sc['num_props'] == 0 and not sc['scaffold']:
        # ---- unconditional: distribution learning ----
        n_iter = math.ceil(a.gen_size / a.batch_size)
        gen = []
        for _ in range(n_iter):
            gen += gen_batch(model, stoi, itos, block_size, a.temp, a.batch_size)
        bm, uniq = basic_metrics(gen, n_iter * a.batch_size, train_set)
        rec.update(bm)
        ref = get_reference(holdout, n=10000)
        uniq_eval = uniq if len(uniq) <= 10000 else list(np.random.RandomState(42).choice(uniq, 10000, replace=False))
        rec.update(distribution_metrics(uniq_eval, ref))
        all_gen = [{'smiles': s} for s in gen]
        rec['mode'] = 'unconditional'

    else:
        # ---- conditional / scaffold ----
        prop_targets = PROP2VALUE['_'.join(sc['props'])] if sc['num_props'] > 0 else [None]
        scaf_list = SCAFFOLDS if sc['scaffold'] else [None]
        n_iter = math.ceil(a.cond_size / a.batch_size)
        per_cond = []
        for j in scaf_list:
            sca_t = None
            if j is not None:
                padded = j + '<' * (scaffold_max_len - len(REGEX.findall(j)))
                sca_t = torch.tensor([stoi[s] for s in REGEX.findall(padded)], dtype=torch.long)[None, ...].repeat(a.batch_size, 1)
            for c in prop_targets:
                p_t = None
                if c is not None:
                    if sc['num_props'] == 1:
                        p_t = torch.tensor([[c]]).repeat(a.batch_size, 1)
                    else:
                        p_t = torch.tensor([c]).repeat(a.batch_size, 1).unsqueeze(1)
                gen = []
                for _ in range(n_iter):
                    gen += gen_batch(model, stoi, itos, block_size, a.temp, a.batch_size, prop=p_t, scaffold=sca_t)
                bm, uniq = basic_metrics(gen, n_iter * a.batch_size, train_set)
                centry = {'condition': c, 'scaffold': j, **bm}
                # MAD per property
                if c is not None:
                    mols = [get_mol(s) for s in gen]
                    mols = [m for m in mols if m is not None]
                    for k, pname in enumerate(sc['props']):
                        tgt = c if sc['num_props'] == 1 else c[k]
                        vals = [PROP_FN[pname](m) for m in mols]
                        centry[f'mad_{pname}'] = round(float(np.mean(np.abs(np.array(vals) - tgt))), 4) if vals else None
                # scaffold-match
                if j is not None:
                    j_canon = canonic_smiles(j)
                    match = 0; tot = 0
                    for s in gen:
                        m = get_mol(s)
                        if m is None:
                            continue
                        tot += 1
                        try:
                            if MurckoScaffoldSmiles(mol=m) == j_canon:
                                match += 1
                        except Exception:
                            pass
                    centry['scaffold_match'] = round(match / tot, 4) if tot else 0.0
                per_cond.append(centry)
                all_gen += [{'smiles': s, 'condition': str(c), 'scaffold': j} for s in gen]
        rec['per_condition'] = per_cond
        rec['mode'] = ('scaffold+prop' if (sc['scaffold'] and sc['num_props'] > 0)
                       else 'scaffold' if sc['scaffold'] else 'conditional')
        # aggregate means
        rec['validity'] = round(float(np.mean([e['validity'] for e in per_cond])), 4)
        rec['unique'] = round(float(np.mean([e['unique'] for e in per_cond])), 4)
        rec['novelty'] = round(float(np.mean([e['novelty'] for e in per_cond])), 4)
        for pname in (sc['props'] if sc['num_props'] > 0 else []):
            mads = [e[f'mad_{pname}'] for e in per_cond if e.get(f'mad_{pname}') is not None]
            rec[f'mad_{pname}'] = round(float(np.mean(mads)), 4) if mads else None
        if sc['scaffold']:
            rec['scaffold_match'] = round(float(np.mean([e['scaffold_match'] for e in per_cond])), 4)

    out_csv = os.path.join(ROOT, 'datasets', f"gen_guaca_{a.run_name}_T{a.temp}.csv")
    pd.DataFrame(all_gen).to_csv(out_csv, index=False)
    out_json = os.path.join(ROOT, 'datasets', f"guaca_metrics_{a.run_name}_T{a.temp}.json")
    with open(out_json, 'w') as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != 'per_condition'}, indent=2))
    print('wrote', out_json)


if __name__ == '__main__':
    main()
