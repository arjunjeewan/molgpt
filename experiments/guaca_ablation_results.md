# GuacaMol component ablation — comprehensive results

### Table A — Component ablation (unconditional GuacaMol, T=1.0)

Each component ✓=on / ✗=off. Metrics are mean over seeds (±std where >1 seed). Validity/Unique/Novelty and KL-div: higher=better. FCD: lower=better (distance to held-out reference).

| Config | RoPE | SwiGLU | RMSNorm | Validity | Unique | Novelty | KL-div | FCD |
|---|:--:|:--:|:--:|--:|--:|--:|--:|--:|
| baseline (3s: 1,2,42) | ✗ | ✗ | ✗ | 0.977±0.001 | 0.999±0.000 | 0.955±0.001 | 0.994±0.001 | 0.992±0.007 |
| +RoPE (3s: 1,2,42) | ✓ | ✗ | ✗ | 0.981±0.001 | 0.999±0.000 | 0.960±0.003 | 0.988±0.001 | 1.253±0.026 |
| +SwiGLU (3s: 1,2,42) | ✗ | ✓ | ✗ | 0.977±0.002 | 0.999±0.000 | 0.947±0.002 | 0.995±0.000 | 1.004±0.009 |
| +RMSNorm (3s: 1,2,42) | ✗ | ✗ | ✓ | 0.979±0.001 | 0.999±0.000 | 0.953±0.001 | 0.994±0.000 | 0.969±0.024 |
| modern (all) (3s: 1,2,42) | ✓ | ✓ | ✓ | 0.981±0.001 | 0.999±0.000 | 0.949±0.002 | 0.989±0.001 | 1.336±0.025 |

### Table B — Generation modes, baseline vs modern (GuacaMol, T=1.0)

Validity/Unique/Novelty: higher=better. MAD (mean abs deviation of the generated property from the target, averaged over the target grid): lower=better. Scaffold-match: fraction of generated mols whose Murcko scaffold equals the conditioning scaffold (higher=better).

| Mode | Arch | Validity | Unique | Novelty | MAD | Scaffold-match |
|---|---|--:|--:|--:|--:|--:|
| prop:logp | baseline | 0.969 | 1.000 | 0.978 | logp=0.224 | - |
| prop:logp | modern | 0.972 | 0.999 | 0.971 | logp=0.200 | - |
| scaffold | baseline | 0.994 | 0.794 | 0.992 | - | 0.973 |
| scaffold | modern | 0.995 | 0.801 | 0.989 | - | 0.982 |
| scaffold+logp | baseline | 0.989 | 0.827 | 0.999 | logp=0.192 | 0.979 |
| scaffold+logp | modern | 0.989 | 0.814 | 0.999 | logp=0.199 | 0.962 |
