"""
GuacaMol training driver for the component ablation. One code path; the architecture is
selected by --config (which sets the RoPE/SwiGLU/RMSNorm toggles on model_ablate.GPT),
and the conditioning by --num_props/--props/--scaffold (mirrors train/train.py's guacamol
branch: 'source' column, hardcoded 94-token vocab).

Writes:
  ../cond_gpt/weights/<run_name>.pt    best-val state_dict
  ../cond_gpt/weights/<run_name>.json  sidecar: everything eval needs to rebuild + generate
"""
import os, sys, re, json, argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # molgpt/
sys.path.insert(0, os.path.join(ROOT, 'train'))   # trainer, dataset, utils
sys.path.insert(0, os.path.join(ROOT, 'experiments'))     # model_ablate
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
from utils import set_seed
from model_ablate import GPT, GPTConfig

CONFIG_MAP = {              # (use_rope, use_swiglu, use_rmsnorm)
    'baseline': (False, False, False),
    'rope':     (True,  False, False),
    'swiglu':   (False, True,  False),
    'rmsnorm':  (False, False, True),
    'modern':   (True,  True,  True),
}

WHOLE_STRING = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class _Wandb:
    def log(self, *a, **k):
        pass


class _Args:
    debug = False


def compute_lens(data_name, smiles_all, scaffold_all):
    """max SMILES / scaffold token length over the full split; cached per dataset."""
    cache = os.path.join(ROOT, 'datasets', f'{data_name}_lens.json')
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        return d['max_len'], d['scaffold_max_len']
    regex = re.compile(PATTERN)
    max_len = max(len(regex.findall(str(i).strip())) for i in smiles_all)
    scaffold_max_len = max(len(regex.findall(str(i).strip())) for i in scaffold_all)
    with open(cache, 'w') as f:
        json.dump({'max_len': max_len, 'scaffold_max_len': scaffold_max_len}, f)
    return max_len, scaffold_max_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=list(CONFIG_MAP), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--data_name', default='guacamol2')
    ap.add_argument('--num_props', type=int, default=0)
    ap.add_argument('--props', nargs='+', default=['qed'])
    ap.add_argument('--scaffold', action='store_true', default=False)
    ap.add_argument('--batch_size', type=int, default=384)
    a = ap.parse_args()

    use_rope, use_swiglu, use_rmsnorm = CONFIG_MAP[a.config]
    set_seed(a.seed)

    data = pd.read_csv(os.path.join(ROOT, 'datasets', f'{a.data_name}.csv'))
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    # guacamol uses 'source' (train/val/test); moses uses 'split'
    split_col = 'split' if 'moses' in a.data_name else 'source'
    train_val = ('train', 'test') if 'moses' in a.data_name else ('train', 'val')
    train_data = data[data[split_col] == train_val[0]].reset_index(drop=True)
    val_data = data[data[split_col] == train_val[1]].reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    prop = train_data[a.props].values.tolist()
    vprop = val_data[a.props].values.tolist()
    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']

    max_len, scaffold_max_len = compute_lens(
        a.data_name,
        list(smiles.values) + list(vsmiles.values),
        list(scaffold.values) + list(vscaffold.values))
    print(f'max_len={max_len} scaffold_max_len={scaffold_max_len}', flush=True)

    regex = re.compile(PATTERN)
    smiles = [i + '<' * (max_len - len(regex.findall(str(i).strip()))) for i in smiles]
    vsmiles = [i + '<' * (max_len - len(regex.findall(str(i).strip()))) for i in vsmiles]
    scaffold = [str(i) + '<' * (scaffold_max_len - len(regex.findall(str(i).strip()))) for i in scaffold]
    vscaffold = [str(i) + '<' * (scaffold_max_len - len(regex.findall(str(i).strip()))) for i in vscaffold]

    args = _Args()
    train_dataset = SmileDataset(args, smiles, WHOLE_STRING, max_len, prop=prop, aug_prob=0,
                                 scaffold=scaffold, scaffold_maxlen=scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, WHOLE_STRING, max_len, prop=vprop, aug_prob=0,
                                 scaffold=vscaffold, scaffold_maxlen=scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=a.num_props,
                      n_layer=8, n_head=8, n_embd=256, scaffold=a.scaffold, scaffold_maxlen=scaffold_max_len,
                      lstm=False, lstm_layers=0,
                      use_rope=use_rope, use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm)
    model = GPT(mconf)
    print(f'[{a.config}] rope={use_rope} swiglu={use_swiglu} rmsnorm={use_rmsnorm} '
          f'seed={a.seed} epochs={a.epochs} num_props={a.num_props} scaffold={a.scaffold} '
          f'params={sum(p.numel() for p in model.parameters())}', flush=True)

    ckpt = os.path.abspath(os.path.join(ROOT, '..', 'cond_gpt', 'weights', f'{a.run_name}.pt'))
    tconf = TrainerConfig(max_epochs=a.epochs, batch_size=a.batch_size, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=a.epochs * len(train_data) * max_len,
                          num_workers=10, ckpt_path=ckpt, block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, train_dataset.stoi, train_dataset.itos)
    trainer.train(_Wandb())

    sidecar = {
        'config': a.config, 'use_rope': use_rope, 'use_swiglu': use_swiglu, 'use_rmsnorm': use_rmsnorm,
        'num_props': a.num_props, 'props': a.props if a.num_props > 0 else [], 'scaffold': a.scaffold,
        'max_len': max_len, 'scaffold_max_len': scaffold_max_len, 'vocab_size': train_dataset.vocab_size,
        'n_layer': 8, 'n_head': 8, 'n_embd': 256, 'data_name': a.data_name, 'seed': a.seed, 'epochs': a.epochs,
    }
    with open(ckpt.replace('.pt', '.json'), 'w') as f:
        json.dump(sidecar, f, indent=2)
    print(f'DONE [{a.config}] seed={a.seed} -> {ckpt}', flush=True)


if __name__ == '__main__':
    main()
