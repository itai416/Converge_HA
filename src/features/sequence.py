"""ESM-2 features of the mutated site, computed on the mutated chain only (frozen model, no partner context).

Per mutation:
  llr        log p(mut) - log p(wt) at the masked site (masked-marginal zero-shot score)
  wt_emb_*   last-layer embedding of the site in the wild-type chain
  diff_emb_* site embedding in the mutant chain minus the wild-type one
"""
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from src.features.geometry import _structure
from src.data.mapping import AA3


def chain_sequence(pdb_id, chain):
    """Observed residues of a chain in the SKEMPI .pdb file, and a map residue number -> sequence index."""
    residues = [r for r in _structure(pdb_id)[chain] if r.id[0] == " " and r.get_resname() in AA3]
    seq = "".join(AA3[r.get_resname()] for r in residues)
    return seq, {r.id[1]: i for i, r in enumerate(residues)}


class ESM2:
    def __init__(self, name="facebook/esm2_t12_35M_UR50D", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = EsmForMaskedLM.from_pretrained(name).eval().to(device)
        self.device = device

    @torch.no_grad()
    def _run(self, seq):
        ids = self.tok(seq, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model(ids, output_hidden_states=True)
        return ids, out.logits[0], out.hidden_states[-1][0]

    @lru_cache(maxsize=None)
    def _wt(self, seq):
        _, _, h = self._run(seq)
        return h.cpu().numpy()

    @torch.no_grad()
    def features(self, seq, idx, wt, mut):
        """idx is 0-based in `seq`; token position is idx + 1 because of the <cls> token."""
        assert seq[idx] == wt, (seq[idx], wt)
        ids = self.tok(seq, return_tensors="pt")["input_ids"].to(self.device)
        ids[0, idx + 1] = self.tok.mask_token_id
        logp = torch.log_softmax(self.model(ids).logits[0, idx + 1], -1)
        llr = (logp[self.tok.convert_tokens_to_ids(mut)] - logp[self.tok.convert_tokens_to_ids(wt)]).item()
        wt_emb = self._wt(seq)[idx + 1]
        _, _, h_mut = self._run(seq[:idx] + mut + seq[idx + 1:])
        return llr, wt_emb, h_mut[idx + 1].cpu().numpy() - wt_emb
