"""ESM-IF1 (inverse folding) features of the mutated site, computed on the wild-type complex structure.

ESM-IF1 encodes the backbone (N, CA, C) of the given chains and decodes a sequence for the *target* chain
autoregressively: p(seq_i | structure, seq_<i). Following fair-esm's multichain recipe, the target chain is placed
first and the other chains follow as structural context.

Two structural contexts per mutation:
  complex  target chain + all chains of both partners (antibody and antigen)
  unbound  target chain + the other chains of its own partner only (e.g. the light chain for a heavy-chain mutation)
The difference complex - unbound isolates what the binding partner contributes: an interface-specific signal.

Per context:
  site_llr  log p(mut) - log p(wt) at the site, from the wild-type pass (conditioned on the native preceding residues)
  seq_llr   log-likelihood of the whole mutant chain minus the wild-type chain (the standard ESM-IF1 mutation score;
            it also counts the effect of the mutation on the residues decoded after it)
  emb       encoder embedding of the site (structure only, so identical for wild type and mutant)
"""
from functools import lru_cache

import biotite.structure as bs
import numpy as np
import torch
from biotite.structure.io import pdb

from src.data.mapping import AA3, PDB_DIR
from src.features.geometry import antibody_chains

PADDING = 10  # residues of NaN coordinates between concatenated chains (fair-esm default)

# fair-esm imports biotite's `filter_backbone`, which biotite 1.x renamed `filter_peptide_backbone` (same selection: N, CA, C
# of amino-acid residues). NumPy 2 environments (e.g. current Colab) need biotite >= 1.0, so restore the old name before
# fair-esm is imported.
if not hasattr(bs, "filter_backbone"):
    bs.filter_backbone = bs.filter_peptide_backbone


def _install_torch_scatter_fallback():
    """fair-esm imports `scatter_add` and `scatter` from torch_scatter, a compiled package whose wheels must match the exact
    torch + CUDA build (often missing right after Colab upgrades torch). It only ever calls `scatter_add`, so when the
    package is absent we register an equivalent pure-PyTorch module instead."""
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("torch_scatter") is not None:
        return

    def scatter_add(src, index, dim=-1, out=None, dim_size=None):
        dim = dim % src.dim()
        if index.dim() == 1 and src.dim() > 1:
            index = index.view([-1 if i == dim else 1 for i in range(src.dim())]).expand_as(src)
        if out is None:
            shape = list(src.shape)
            shape[dim] = dim_size if dim_size is not None else int(index.max()) + 1
            out = src.new_zeros(shape)
        return out.scatter_add_(dim, index, src)

    def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        if reduce not in ("sum", "add"):
            raise NotImplementedError(f"torch_scatter fallback: reduce={reduce!r}")
        return scatter_add(src, index, dim, out, dim_size)

    module = types.ModuleType("torch_scatter")
    module.scatter_add, module.scatter = scatter_add, scatter
    sys.modules["torch_scatter"] = module


_install_torch_scatter_fallback()


@lru_cache(maxsize=None)
def load_chains(pdb_id):
    """{chain: (N/CA/C coords L x 3 x 3, sequence, residue numbers)} from the SKEMPI .pdb file."""
    from esm.inverse_folding.util import get_atom_coords_residuewise

    st = pdb.PDBFile.read(str(PDB_DIR / f"{pdb_id}.pdb")).get_structure(model=1)
    st = st[bs.filter_backbone(st)]
    out = {}
    for c in bs.get_chains(st):
        s = st[st.chain_id == c]
        coords = get_atom_coords_residuewise(["N", "CA", "C"], s).astype(np.float32)
        res_ids, res_names = bs.get_residues(s)
        out[c] = (coords, "".join(AA3.get(n, "X") for n in res_names), [int(r) for r in res_ids])
    return out


def contexts(complex_id, chain):
    """Chain order for the two contexts: target chain first, then context chains."""
    ab, ag = antibody_chains(complex_id)
    present = load_chains(complex_id[:4])
    own = ab if chain in ab else ag
    complex_chains = [chain] + [c for c in ab + ag if c != chain and c in present]
    unbound_chains = [chain] + [c for c in own if c != chain and c in present]
    return {"complex": complex_chains, "unbound": unbound_chains}


def _concat(pdb_id, chains):
    data = load_chains(pdb_id)
    pad = np.full((PADDING, 3, 3), np.nan, dtype=np.float32)
    parts = [data[chains[0]][0]]
    for c in chains[1:]:
        parts += [pad, data[c][0]]
    return np.concatenate(parts, axis=0)


class ESMIF1:
    def __init__(self, device=None):
        import esm
        from esm.inverse_folding.util import CoordBatchConverter

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model = self.model.eval().to(self.device)
        self.bc = CoordBatchConverter(self.alphabet)

    def _batch(self, coords, seq):
        return self.bc([(coords, None, seq)], device=self.device)

    @torch.no_grad()
    def logprobs(self, coords, seq):
        """(len(seq) + 1) x vocab log-probabilities for the target sequence (last row = end token).

        Same computation as fair-esm's get_sequence_loss, but on self.device (that function builds CPU tensors only).
        """
        c, conf, _, tokens, pad = self._batch(coords, seq)
        logits, _ = self.model.forward(c, pad, conf, tokens[:, :-1])
        return torch.log_softmax(logits[0].float(), dim=0).T.cpu().numpy(), tokens[0, 1:].cpu().numpy()

    def seq_loglik(self, coords, seq):
        lp, target = self.logprobs(coords, seq)
        keep = target != self.alphabet.padding_idx
        return float(lp[np.arange(len(target)), target][keep].sum())

    @torch.no_grad()
    def encoder_out(self, coords):
        c, conf, _, _, pad = self._batch(coords, None)
        out = self.model.encoder.forward(c, pad, conf, return_all_hiddens=False)
        return out["encoder_out"][0][1:-1, 0].float().cpu().numpy()  # drop begin/end tokens

    @lru_cache(maxsize=None)
    def _wild_type(self, pdb_id, chains):
        coords = _concat(pdb_id, list(chains))
        seq = load_chains(pdb_id)[chains[0]][1]
        lp, target = self.logprobs(coords, seq)
        keep = target != self.alphabet.padding_idx
        ll = float(lp[np.arange(len(target)), target][keep].sum())
        return coords, seq, lp, ll, self.encoder_out(coords)

    def features(self, complex_id, chain, resnum, wt, mut):
        pdb_id = complex_id[:4]
        _, seq, resids = load_chains(pdb_id)[chain]
        idx = resids.index(resnum)
        if seq[idx] != wt:
            raise ValueError(f"{complex_id} {chain}{resnum}: structure has {seq[idx]}, mutation says {wt}")
        mut_seq = seq[:idx] + mut + seq[idx + 1:]
        a = self.alphabet
        f, emb = {}, {}
        for ctx, chains in contexts(complex_id, chain).items():
            coords, _, lp, ll_wt, enc = self._wild_type(pdb_id, tuple(chains))
            f[f"if_site_llr_{ctx}"] = float(lp[idx, a.get_idx(mut)] - lp[idx, a.get_idx(wt)])
            f[f"if_seq_llr_{ctx}"] = self.seq_loglik(coords, mut_seq) - ll_wt
            emb[ctx] = enc[idx]
        for k in ["site_llr", "seq_llr"]:
            f[f"if_{k}_interface"] = f[f"if_{k}_complex"] - f[f"if_{k}_unbound"]
        return f, emb
