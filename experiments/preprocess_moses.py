"""
Reconstruct the molgpt `moses2.csv` from the canonical MOSES dataset bundled in
`molsets`, computing the scaffold + property columns the molgpt code expects.
Columns: smiles, split, scaffold_smiles, qed, sas, logp, tpsa   (lowercase)
"""
import os, sys, time
import pandas as pd
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # RDKit contrib SA scorer (ships fpscores.pkl.gz)

RDLogger.DisableLog('rdApp.*')
import moses


def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaf = ''
    return (smi, scaf, QED.qed(mol), sascorer.calculateScore(mol),
            Crippen.MolLogP(mol), CalcTPSA(mol))


def main():
    splits = [('train', 'train'), ('test', 'test'), ('test_scaffolds', 'test_scaffolds')]
    rows = []
    with Pool(36) as pool:
        for split_key, split_label in splits:
            smis = moses.get_dataset(split_key)
            t0 = time.time()
            out = pool.map(featurize, smis, chunksize=2000)
            ok = [r for r in out if r is not None]
            for smi, scaf, qed, sas, logp, tpsa in ok:
                rows.append((smi, split_label, scaf, qed, sas, logp, tpsa))
            print(f"{split_label:15s} {len(ok):>8}/{len(smis)} done in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows, columns=['smiles', 'split', 'scaffold_smiles',
                                     'qed', 'sas', 'logp', 'tpsa'])
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'moses2.csv')
    df.to_csv(out_path, index=False)
    print("wrote", out_path, "rows:", len(df))
    print(df.head(3).to_string())
    print("split counts:\n", df['split'].value_counts().to_string())


if __name__ == '__main__':
    main()
