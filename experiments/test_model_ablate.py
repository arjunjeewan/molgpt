"""
Prove the unified toggle model is equivalent to the two reference implementations:
  all-off  (rope=F,swiglu=F,rmsnorm=F)  ==  experiments/model_baseline.py
  all-on   (rope=T,swiglu=T,rmsnorm=T)  ==  train/model.py
For each, copy the reference's weights into the ablate model (strict state_dict load)
and assert identical logits across all 4 generation modes. Also sanity-check that the
mixed configs build/forward and that all-off has +pos_emb while all-on drops it.
"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # molgpt/
sys.path.insert(0, os.path.join(ROOT, 'experiments'))     # model_baseline, model_ablate
sys.path.insert(0, os.path.join(ROOT, 'train'))   # model (modern)

import model_ablate
import model_baseline
import model as model_modern   # train/model.py (RoPE+SwiGLU+RMSNorm)

torch.manual_seed(0)
VOCAB, BLOCK, SCAF = 94, 100, 98
N_LAYER, N_HEAD, N_EMBD = 8, 8, 256
MODES = [
    ('unconditional', 0, False),
    ('conditional-1prop', 1, False),
    ('scaffold', 0, True),
    ('scaffold+prop', 1, True),
]


def cfg(GPTConfig, num_props, scaffold, **toggles):
    return GPTConfig(VOCAB, BLOCK, num_props=num_props, n_layer=N_LAYER, n_head=N_HEAD,
                     n_embd=N_EMBD, scaffold=scaffold, scaffold_maxlen=SCAF,
                     lstm=False, lstm_layers=0, **toggles)


def make_inputs(num_props, scaffold, B=4, T=20):
    g = torch.Generator().manual_seed(123)
    idx = torch.randint(0, VOCAB, (B, T), generator=g)
    prop = torch.rand(B, num_props, generator=g) if num_props else None
    sca = torch.randint(0, VOCAB, (B, SCAF), generator=g) if scaffold else None
    return idx, prop, sca


def compare(ref_module, toggles, label):
    print(f'\n===== {label}  toggles={toggles} =====')
    all_ok = True
    for name, num_props, scaffold in MODES:
        ref = ref_module.GPT(cfg(ref_module.GPTConfig, num_props, scaffold))
        abl = model_ablate.GPT(cfg(model_ablate.GPTConfig, num_props, scaffold, **toggles))

        np_ref = sum(p.numel() for p in ref.parameters())
        np_abl = sum(p.numel() for p in abl.parameters())

        # weight transfer must be strict (identical module structure / keys)
        missing, unexpected = abl.load_state_dict(ref.state_dict(), strict=False)
        key_ok = (len(missing) == 0 and len(unexpected) == 0)

        ref.eval(); abl.eval()
        idx, prop, sca = make_inputs(num_props, scaffold)
        with torch.no_grad():
            lr, _, _ = ref(idx, prop=prop, scaffold=sca)
            la, _, _ = abl(idx, prop=prop, scaffold=sca)
        same_shape = lr.shape == la.shape
        max_diff = (lr - la).abs().max().item() if same_shape else float('nan')
        logits_ok = same_shape and max_diff < 1e-5
        ok = key_ok and logits_ok and (np_ref == np_abl)
        all_ok &= ok
        print(f'  {name:18s} params ref={np_ref} abl={np_abl} '
              f'| keys miss={len(missing)} unexp={len(unexpected)} '
              f'| logits {tuple(lr.shape)} max|d|={max_diff:.2e} | {"OK" if ok else "FAIL"}')
        if missing:
            print('     missing:', missing[:6])
        if unexpected:
            print('     unexpected:', unexpected[:6])
    return all_ok


def check_pos_emb():
    print('\n===== pos_emb presence =====')
    off = model_ablate.GPT(cfg(model_ablate.GPTConfig, 0, False, use_rope=False, use_swiglu=False, use_rmsnorm=False))
    on = model_ablate.GPT(cfg(model_ablate.GPTConfig, 0, False, use_rope=True, use_swiglu=True, use_rmsnorm=True))
    off_has = any(k == 'pos_emb' for k in off.state_dict())
    on_has = any(k == 'pos_emb' for k in on.state_dict())
    print(f'  all-off has pos_emb: {off_has} (expect True)')
    print(f'  all-on  has pos_emb: {on_has} (expect False)')
    return off_has and not on_has


def check_optimizer():
    print('\n===== configure_optimizers covers all params (all 5 configs) =====')
    class TC:  # minimal train_config
        weight_decay = 0.1; learning_rate = 6e-4; betas = (0.9, 0.95)
    configs = {'baseline': (False, False, False), '+rope': (True, False, False),
               '+swiglu': (False, True, False), '+rmsnorm': (False, False, True),
               'modern': (True, True, True)}
    ok = True
    for name, (r, s, n) in configs.items():
        m = model_ablate.GPT(cfg(model_ablate.GPTConfig, 0, False, use_rope=r, use_swiglu=s, use_rmsnorm=n))
        try:
            m.configure_optimizers(TC())
            print(f'  {name:10s} OK')
        except AssertionError as e:
            ok = False
            print(f'  {name:10s} FAIL: {e}')
    return ok


if __name__ == '__main__':
    a = compare(model_baseline, dict(use_rope=False, use_swiglu=False, use_rmsnorm=False),
                'ablate(all-off)  vs  model_baseline')
    b = compare(model_modern, dict(use_rope=True, use_swiglu=True, use_rmsnorm=True),
                'ablate(all-on)   vs  model.py (modern)')
    c = check_pos_emb()
    d = check_optimizer()
    print('\n=================== SUMMARY ===================')
    print('all-off == baseline :', a)
    print('all-on  == modern   :', b)
    print('pos_emb toggle       :', c)
    print('optimizer coverage   :', d)
    print('ALL PASS:', a and b and c and d)
