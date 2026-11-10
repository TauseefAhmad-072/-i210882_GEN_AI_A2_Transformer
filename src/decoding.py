"""Tasks 3 & 5: explicit KV-cached greedy decoding, uncached reference, greedy speculative decoding.

All decoders call the *bare transformer* (model.model) + lm_head ourselves and pass the
positions, the attention mask and the cache object explicitly:
  * positions : position_ids = arange(past_len, past_len + m)   (RoPE needs absolute positions)
  * mask      : ones(1, past_len + m); the causal part is applied inside attention for the m new queries
  * cache     : transformers.DynamicCache; keys/values of every layer are appended on each forward and
                `crop(n)` rolls the cache back (used for speculative rejection).
Batch size is 1 (no padding), which keeps position handling exact.
"""
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@dataclass
class GenResult:
    tokens: list
    prefill_s: float = 0.0
    decode_s: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def total_s(self):
        return self.prefill_s + self.decode_s

    @property
    def tok_per_s(self):          # end-to-end
        return len(self.tokens) / max(self.total_s, 1e-9)

    @property
    def decode_tok_per_s(self):   # excludes prefill (first token comes from prefill)
        return max(len(self.tokens) - 1, 0) / max(self.decode_s, 1e-9)


def _step(model, ids, cache=None, last_only=False):
    """One explicit forward pass. Returns (logits, cache)."""
    past = cache.get_seq_length() if cache is not None else 0
    m = ids.size(1)
    pos = torch.arange(past, past + m, device=ids.device).unsqueeze(0)
    mask = torch.ones(1, past + m, dtype=torch.long, device=ids.device)
    out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      past_key_values=cache, use_cache=cache is not None)
    h = out.last_hidden_state
    if last_only:
        h = h[:, -1:]
    return model.lm_head(h), cache


def _is_eos(tok, eos_id):
    return eos_id is not None and tok == eos_id


@torch.no_grad()
def greedy_nocache(model, input_ids, max_new_tokens, eos_id=None):
    """Reference: recompute the whole sequence at every step (O(n^2) work)."""
    dev, seq, toks = input_ids.device, input_ids, []
    t0 = time.perf_counter()
    for i in range(max_new_tokens):
        lg, _ = _step(model, seq, None, last_only=True)
        nxt = lg[:, -1].argmax(-1, keepdim=True)
        if i == 0:
            sync(dev); t1 = time.perf_counter()
        toks.append(nxt.item())
        seq = torch.cat([seq, nxt], dim=1)
        if _is_eos(toks[-1], eos_id):
            break
    sync(dev); t2 = time.perf_counter()
    return GenResult(toks, t1 - t0, t2 - t1)


@torch.no_grad()
def greedy_kvcache(model, input_ids, max_new_tokens, eos_id=None):
    """Prefill once, then feed ONE token per step while reusing/updating the cache."""
    dev = input_ids.device
    cache = DynamicCache()
    t0 = time.perf_counter()
    lg, cache = _step(model, input_ids, cache, last_only=True)          # prefill: n tokens
    nxt = lg[:, -1].argmax(-1, keepdim=True)
    sync(dev); t1 = time.perf_counter()
    toks = [nxt.item()]
    while len(toks) < max_new_tokens and not _is_eos(toks[-1], eos_id):
        lg, cache = _step(model, nxt, cache)                            # decode: 1 token, cache grows by 1
        nxt = lg[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt.item())
    sync(dev); t2 = time.perf_counter()
    return GenResult(toks, t1 - t0, t2 - t1, {"final_cache_len": cache.get_seq_length()})


@torch.no_grad()
def speculative_greedy(target, draft, input_ids, max_new_tokens, eos_id=None, k=4):
    """Greedy speculative decoding (output identical to greedy decoding of `target`).

    Invariant at the top of each round (n = current sequence length):
        both caches hold seq[:n-1]; `last` = seq[n-1] is the not-yet-cached pending token.
    Round: draft proposes d1..dk greedily -> target scores [last,d1..dk] in ONE pass ->
    accept the longest prefix with d_j == target_argmax_j, then append target's own token at the first
    mismatch (or the bonus token if all k matched). Roll both caches back to hold the accepted prefix.
    """
    dev = input_ids.device
    ct, cd = DynamicCache(), DynamicCache()
    t0 = time.perf_counter()
    if input_ids.size(1) > 1:
        _step(target, input_ids[:, :-1], ct, last_only=True)
        _step(draft, input_ids[:, :-1], cd, last_only=True)
    last = input_ids[:, -1:]
    sync(dev); t1 = time.perf_counter()
    out, st = [], {"rounds": 0, "drafted": 0, "accepted": 0, "target_calls": 0, "draft_calls": 0}
    done = False
    while not done:
        kk = max(1, min(k, max_new_tokens - len(out)))
        drafts, cur = [], last
        for _ in range(kk):
            lg, cd = _step(draft, cur, cd)
            cur = lg[:, -1].argmax(-1, keepdim=True)
            drafts.append(cur)
        st["draft_calls"] += kk
        d = torch.cat(drafts, dim=1)                                   # (1,kk)
        lg, ct = _step(target, torch.cat([last, d], dim=1), ct)        # (1,kk+1,V) one verify pass
        st["target_calls"] += 1
        g = lg.argmax(-1)                                              # target greedy choice after each input
        a = int((g[:, :kk] == d).long().cumprod(dim=1).sum())          # accepted prefix length
        new = d[0, :a].tolist() + [int(g[0, a])]                       # + correction (or bonus) token
        base = ct.get_seq_length() - (kk + 1)                          # = n-1 before this round
        ct.crop(base + a + 1)                                          # keep last,d1..da ; drop rejected
        if a == kk:                                                    # d_kk was never fed to the draft
            _, cd = _step(draft, d[:, -1:], cd); st["draft_calls"] += 1
        else:
            cd.crop(base + a + 1)
        st["rounds"] += 1; st["drafted"] += kk; st["accepted"] += a
        for t in new:
            out.append(t)
            if _is_eos(t, eos_id) or len(out) >= max_new_tokens:
                done = True
                break
        last = torch.tensor([[out[-1]]], device=dev)
    sync(dev); t2 = time.perf_counter()
    st["acceptance_rate"] = st["accepted"] / max(st["drafted"], 1)
    st["tokens_per_target_call"] = len(out) / max(st["target_calls"], 1)
    return GenResult(out, t1 - t0, t2 - t1, st)


# ---- library references (for correctness comparison only) -----------------------------------
@torch.no_grad()
def hf_generate(model, input_ids, max_new_tokens, eos_id=None, assistant=None):
    dev = input_ids.device
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
              repetition_penalty=1.0, pad_token_id=eos_id or 0)
    if eos_id is None:
        kw["min_new_tokens"] = max_new_tokens
    if assistant is not None:
        kw["assistant_model"] = assistant
    sync(dev); t = time.perf_counter()
    out = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), **kw)
    sync(dev)
    return GenResult(out[0, input_ids.size(1):].tolist(), 0.0, time.perf_counter() - t)


MODES = ["no_cache", "kv_cache", "speculative", "hf_generate", "hf_assisted"]


def generate(mode, model, input_ids, max_new_tokens, eos_id=None, draft=None, k=4):
    if mode == "no_cache":
        return greedy_nocache(model, input_ids, max_new_tokens, eos_id)
    if mode == "kv_cache":
        return greedy_kvcache(model, input_ids, max_new_tokens, eos_id)
    if mode == "speculative":
        assert draft is not None, "speculative mode needs a draft model"
        return speculative_greedy(model, draft, input_ids, max_new_tokens, eos_id, k)
    if mode == "hf_generate":
        return hf_generate(model, input_ids, max_new_tokens, eos_id)
    if mode == "hf_assisted":
        assert draft is not None
        return hf_generate(model, input_ids, max_new_tokens, eos_id, assistant=draft)
    raise ValueError(mode)
