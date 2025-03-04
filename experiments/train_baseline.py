"""
Train the BASELINE (original) MolGPT architecture on the identical moses2.csv with
identical hyperparameters as the modernized run, for an apples-to-apples ablation.
Mirrors train/train.py's unconditional MOSES setup exactly; only the model differs.
"""
import os, sys, re
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # molgpt/
sys.path.insert(0, os.path.join(ROOT, 'experiments'))    # model_baseline
sys.path.insert(0, os.path.join(ROOT, 'train'))  # trainer, dataset, utils
from model_baseline import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
from utils import set_seed


class _Wandb:   # stub: Trainer only calls .log(...)
    def log(self, *a, **k):
        pass


class _Args:
    debug = False


def main():
    set_seed(42)
    run_name = 'unconditional_moses_baseline'

    data = pd.read_csv(os.path.join(ROOT, 'datasets', 'moses2.csv'))
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    train_data = data[data['split'] == 'train'].reset_index(drop=True)
    val_data = data[data['split'] == 'test'].reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    prop = train_data[['qed']].values.tolist()
    vprop = val_data[['qed']].values.tolist()
    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    max_len = max(len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values)))
    scaffold_max_len = max(len(regex.findall(str(i).strip())) for i in (list(scaffold.values) + list(vscaffold.values)))
    print('Max len:', max_len, 'Scaffold max len:', scaffold_max_len, flush=True)

    smiles = [i + '<' * (max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + '<' * (max_len - len(regex.findall(i.strip()))) for i in vsmiles]
    scaffold = [str(i) + '<' * (scaffold_max_len - len(regex.findall(str(i).strip()))) for i in scaffold]
    vscaffold = [str(i) + '<' * (scaffold_max_len - len(regex.findall(str(i).strip()))) for i in vscaffold]

    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']

    args = _Args()
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen=scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen=scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=0,
                      n_layer=8, n_head=8, n_embd=256, scaffold=False, scaffold_maxlen=scaffold_max_len,
                      lstm=False, lstm_layers=0)
    model = GPT(mconf)
    print('baseline params:', sum(p.numel() for p in model.parameters()), flush=True)

    ckpt = os.path.join(ROOT, '..', 'cond_gpt', 'weights', f'{run_name}.pt')
    tconf = TrainerConfig(max_epochs=10, batch_size=384, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=10 * len(train_data) * max_len,
                          num_workers=10, ckpt_path=os.path.abspath(ckpt),
                          block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, train_dataset.stoi, train_dataset.itos)
    trainer.train(_Wandb())
    print('DONE baseline training ->', os.path.abspath(ckpt), flush=True)


if __name__ == '__main__':
    main()
