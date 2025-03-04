"""
Generate 10k molecules from the trained (modified) MolGPT unconditional MOSES model
and compute MOSES metrics, for comparison against Table 1 of the manuscript.
Mirrors train.py's data prep so the model config (and mask buffer shapes) match the
saved checkpoint exactly.
"""
import os, sys, re, math, argparse
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # molgpt/
sys.path.insert(0, os.path.join(ROOT, 'generate')) # sample(), get_mol, model
from utils import sample
from get_mol import get_mol
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import moses

# exact vocab list from train/train.py (line 123)
WHOLE_STRING = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']

PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--baseline', action='store_true', help='use original architecture (model_baseline)')
    ap.add_argument('--temp', type=float, default=1.6)
    ap.add_argument('--gen_size', type=int, default=10000)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--tag', default=None, help='filename tag for per-seed outputs (default: baseline/modified)')
    args = ap.parse_args()

    if args.baseline:
        sys.path.insert(0, HERE)
        from model_baseline import GPT, GPTConfig
        default_ckpt = os.path.join(ROOT, '..', 'cond_gpt', 'weights', 'unconditional_moses_baseline.pt')
        tag = 'BASELINE (original arch)'
    else:
        from model import GPT, GPTConfig
        default_ckpt = os.path.join(ROOT, '..', 'cond_gpt', 'weights', 'unconditional_moses.pt')
        tag = 'modified (RoPE+SwiGLU+RMSNorm)'
    if args.ckpt is None:
        args.ckpt = default_ckpt

    regex = re.compile(PATTERN)
    data = pd.read_csv(os.path.join(ROOT, 'datasets', 'moses2.csv'))
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    train_smiles = data[data['split'] == 'train']['smiles']
    val_smiles = data[data['split'] == 'test']['smiles']
    train_scaf = data[data['split'] == 'train']['scaffold_smiles']
    val_scaf = data[data['split'] == 'test']['scaffold_smiles']

    max_len = max(len(regex.findall(s.strip())) for s in list(train_smiles.values) + list(val_smiles.values))
    scaffold_max_len = max(len(regex.findall(str(s).strip())) for s in list(train_scaf.values) + list(val_scaf.values))
    print('max_len', max_len, 'scaffold_max_len', scaffold_max_len)

    chars = sorted(set(WHOLE_STRING))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    print('vocab_size', vocab_size)

    mconf = GPTConfig(vocab_size, max_len, num_props=0, n_layer=8, n_head=8, n_embd=256,
                      scaffold=False, scaffold_maxlen=scaffold_max_len, lstm=False, lstm_layers=0)
    model = GPT(mconf)
    sd = torch.load(os.path.abspath(args.ckpt), map_location='cpu')
    model.load_state_dict(sd)
    model.to('cuda').eval()
    print('loaded', os.path.abspath(args.ckpt))

    context = 'C'
    n_iter = math.ceil(args.gen_size / args.batch_size)
    gen_smiles = []
    x0 = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...]
    for _ in range(n_iter):
        x = x0.repeat(args.batch_size, 1).to('cuda')
        y = sample(model, x, max_len, temperature=args.temp, sample=True, top_k=None, prop=None, scaffold=None)
        for row in y:
            completion = ''.join(itos[int(i)] for i in row).replace('<', '')
            gen_smiles.append(completion)
    print('generated', len(gen_smiles), 'raw strings at temperature', args.temp)

    metrics = moses.get_all_metrics(gen_smiles, n_jobs=8, device='cuda', batch_size=512)
    print('\n===== MOSES metrics [%s] unconditional, T=%.1f =====' % (tag, args.temp))
    for k, v in metrics.items():
        print(f'  {k:18s} {v:.4f}')

    # save raw generations
    suffix = args.tag or ('baseline' if args.baseline else 'modified')
    out = os.path.join(ROOT, 'datasets', f'gen_uncond_moses_{suffix}_T{args.temp}.csv')
    pd.DataFrame({'smiles': gen_smiles}).to_csv(out, index=False)
    print('wrote', out)

    # save structured metrics for cross-seed aggregation
    import json
    rec = {'tag': suffix, 'arch': 'baseline' if args.baseline else 'modern', 'temp': args.temp}
    rec.update({k: float(v) for k, v in metrics.items()})
    mout = os.path.join(ROOT, 'datasets', f'moses_metrics_{suffix}_T{args.temp}.json')
    with open(mout, 'w') as f:
        json.dump(rec, f, indent=2)
    print('wrote', mout)


if __name__ == '__main__':
    main()
