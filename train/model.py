"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def rotate_half(x):
    """ rotates half the hidden dims of the input (GPT-NeoX / Llama style). """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, n_head, S, head_dim); cos, sin: (S, head_dim)
    cos = cos[None, None, :, :].to(dtype=q.dtype)
    sin = sin[None, None, :, :].to(dtype=q.dtype)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    """ Rotary Position Embedding (RoPE). Injects position by rotating Q/K vectors. """

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (seq_len, dim)
        return emb.cos(), emb.sin()


class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization (no mean-centering, no bias). """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_props)) + int(config.scaffold_maxlen)   #int(config.lstm_layers)    #  int(config.scaffold) 
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head
        # rotary position embedding applied per-head to query/key vectors
        self.rotary = RotaryEmbedding(config.n_embd // config.n_head)

    def forward(self, x, num_cond=0, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply rotary position embeddings to the SMILES tokens only (positions num_cond..T-1),
        # leaving the prepended condition tokens (property / scaffold) unrotated.
        if T - num_cond > 0:
            cos, sin = self.rotary(T - num_cond, x.device)
            q_s, k_s = apply_rotary_pos_emb(q[:, :, num_cond:, :], k[:, :, num_cond:, :], cos, sin)
            q = torch.cat([q[:, :, :num_cond, :], q_s], dim=2)
            k = torch.cat([k[:, :, :num_cond, :], k_s], dim=2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class SwiGLU(nn.Module):
    """ SwiGLU feed-forward block: down( SiLU(gate(x)) * up(x) ).
    Hidden width is set to ~2/3 * 4*n_embd so the 3 projections keep the
    parameter count close to the original 2-matrix GELU FFN. """

    def __init__(self, config):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config)

    def forward(self, x, num_cond=0):
        y, attn = self.attn(self.ln1(x), num_cond=num_cond)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)
     
        # positional information is injected via Rotary Position Embeddings (RoPE) inside
        # each attention layer, so no learned absolute position embedding is used.
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

        if config.lstm:
            self.lstm = nn.LSTM(input_size = config.n_embd, hidden_size = config.n_embd, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, prop = None, scaffold = None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.config.num_props:
            assert prop.size(-1) == self.config.num_props, "Num_props should be equal to last dim of property vector"           

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device))
        x = self.drop(token_embeddings + type_embeddings)

        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            if prop.ndim == 2:
                p = self.prop_nn(prop.unsqueeze(1))    # for single property
            else:
                p = self.prop_nn(prop)    # for multiproperty
            p += type_embd
            x = torch.cat([p, x], 1)

        if self.config.scaffold:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))

            scaffold_embeds = self.tok_emb(scaffold)     # .mean(1, keepdim = True)
            if self.config.lstm:
                scaffold_embeds = self.lstm(scaffold_embeds.permute(1,0,2))[1][0]
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], 2, self.config.n_embd).mean(2)
                scaffold_embeds = scaffold_embeds.permute(1,0,2)   # mean(0, keepdim = True)
                # scaffold_embeds = scaffold_embeds.reshape(self.config.lstm_layers, 1, -1, self.config.n_embd)[-1].permute(1,0,2)
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], self.config.n_embd)
            scaffold_embeds += type_embd
            x = torch.cat([scaffold_embeds, x], 1)

        # x = self.blocks(x)
        attn_maps = []

        # number of prepended condition tokens (property / scaffold); SMILES tokens follow.
        num_cond = x.size(1) - t

        for layer in self.blocks:
            x, attn = layer(x, num_cond)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        # print(logits.shape)
        if self.config.num_props and self.config.scaffold:
            num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)
        elif self.config.num_props:
            num = int(bool(self.config.num_props))
        elif self.config.scaffold:
            num = int(self.config.scaffold_maxlen) 
        else:
            num = 0

        logits = logits[:, num:, :]


        # if self.config.num_props or self.config.scaffold:

        #     num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)  #int(self.config.lstm_layers)   # int(self.config.scaffold)      # int(self.config.scaffold)
            

        # print(logits.shape)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)