"""
Decoding sweep (NO retraining): vary sampling temperature and nucleus (top-p) on a
trained unconditional-MOSES checkpoint, to test whether the ~5pt validity shortfall
vs the manuscript is a *decoding* artifact (we sampled at T=1.6) rather than the
architecture. Reuses gen_eval_moses.py's checkpoint-matching data prep, but adds a
local sampler with top-p and computes only the FAST metrics (validity / uniqueness /
novelty / internal-diversity) so many configs can be scanned cheaply.
"""
import os, sys, re, math, argparse
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # molgpt/
sys.path.insert(0, os.path.join(ROOT, 'generate'))  # get_mol, model
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from moses.metrics.metrics import fraction_valid, fraction_unique, novelty, internal_diversity

WHOLE_STRING = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

# (temperature, top_p)  -- top_p=None means pure temperature sampling
GRID = [
    (0.7, None), (0.8, None), (0.9, None), (1.0, None), (1.2, None), (1.6, None),
    (1.0, 0.95), (1.0, 0.90), (1.2, 0.95),
]


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def sample_seq(model, x, steps, temperature=1.0, top_p=None):
    block_size = model.get_block_size()
    model.eval()
    for _ in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _, _ = model(x_cond, prop=None, scaffold=None)
        logits = logits[:, -1, :] / temperature
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[:, 1:] = remove[:, :-1].clone()   # keep the first token over the threshold
            remove[:, 0] = False
            remove = remove.scatter(1, sorted_idx, remove)
            logits = logits.masked_fill(remove, -float('Inf'))
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    return x


def generate(model, stoi, itos, regex, max_len, n, batch_size, temperature, top_p):
    x0 = torch.tensor([stoi[s] for s in regex.findall('C')], dtype=torch.long)[None, ...]
    out = []
    for _ in range(math.ceil(n / batch_size)):
        x = x0.repeat(batch_size, 1).to('cuda')
        y = sample_seq(model, x, max_len, temperature=temperature, top_p=top_p)
        for row in y:
            out.append(''.join(itos[int(i)] for i in row).replace('<', ''))
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--gen_size', type=int, default=10000)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--intdiv_n', type=int, default=5000, help='subsample size for IntDiv (cost bound)')
    ap.add_argument('--temps', default=None,
                    help='space-separated temperatures for a pure-temperature frontier; '
                         'overrides the default (temp,top_p) GRID')
    ap.add_argument('--out', default=None, help='explicit output CSV path (per-seed runs)')
    args = ap.parse_args()

    global GRID
    if args.temps is not None:
        GRID = [(float(t), None) for t in args.temps.split()]

    if args.baseline:
        sys.path.insert(0, HERE)
        from model_baseline import GPT, GPTConfig
        default_ckpt = os.path.join(ROOT, '..', 'cond_gpt', 'weights', 'unconditional_moses_baseline.pt')
        tag = 'BASELINE'
    else:
        from model import GPT, GPTConfig
        default_ckpt = os.path.join(ROOT, '..', 'cond_gpt', 'weights', 'unconditional_moses.pt')
        tag = 'MODERNIZED'
    ckpt = args.ckpt or default_ckpt

    regex = re.compile(PATTERN)
    data = pd.read_csv(os.path.join(ROOT, 'datasets', 'moses2.csv')).dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    train_smiles = data[data['split'] == 'train']['smiles']
    val_smiles = data[data['split'] == 'test']['smiles']
    max_len = max(len(regex.findall(s.strip())) for s in list(train_smiles.values) + list(val_smiles.values))
    train_list = train_smiles.tolist()   # canonical MOSES train -> novelty reference

    chars = sorted(set(WHOLE_STRING))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    mconf = GPTConfig(len(chars), max_len, num_props=0, n_layer=8, n_head=8, n_embd=256,
                      scaffold=False, scaffold_maxlen=48, lstm=False, lstm_layers=0)
    model = GPT(mconf)
    model.load_state_dict(torch.load(os.path.abspath(ckpt), map_location='cpu'))
    model.to('cuda').eval()
    print(f'loaded [{tag}] {os.path.abspath(ckpt)}  max_len={max_len}  gen_size={args.gen_size}')

    rows = []
    for temp, top_p in GRID:
        set_seed(42)
        gen = generate(model, stoi, itos, regex, max_len, args.gen_size, args.batch_size, temp, top_p)
        valid = fraction_valid(gen, n_jobs=8)
        canon = [Chem.MolToSmiles(m) for m in (Chem.MolFromSmiles(s) for s in gen) if m is not None]
        uniq = len(set(canon)) / len(canon) if canon else 0.0
        nov = novelty(gen, train_list, n_jobs=8)
        sub = canon if len(canon) <= args.intdiv_n else list(np.random.RandomState(0).choice(canon, args.intdiv_n, replace=False))
        intdiv = internal_diversity(sub, n_jobs=8, device='cuda') if sub else 0.0
        rows.append(dict(temp=temp, top_p=top_p, valid=valid, unique=uniq, novelty=nov, intdiv1=float(intdiv)))
        print(f'  T={temp:<3} top_p={str(top_p):<5}  valid={valid:.4f}  unique={uniq:.4f}  '
              f'novelty={nov:.4f}  IntDiv1={float(intdiv):.4f}')

    df = pd.DataFrame(rows)
    print('\n===== decoding sweep [%s] =====' % tag)
    print(df.to_string(index=False))
    out = args.out or os.path.join(ROOT, 'datasets', f'sweep_decode_{"baseline" if args.baseline else "modified"}.csv')
    df.to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
