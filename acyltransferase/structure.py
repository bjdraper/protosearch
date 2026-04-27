"""AlphaFold DB download + ESMFold local prediction + centroid selection."""

from __future__ import annotations
import json
import pathlib
from pathlib import Path
from typing import Optional

import numpy as np


# ── AlphaFold DB ──────────────────────────────────────────────────────────────

def download_alphafold(uniprot_acc: str, output_dir: str | Path,
                       name: str = "") -> Optional[Path]:
    """
    Download AlphaFold structure for a UniProt accession.
    Queries EBI API to get the real pdbUrl (handles version changes).
    Returns path to saved PDB or None if not found.
    """
    from .utils import http_get
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # check for any existing version
    existing = list(output_dir.glob(f"AF-{uniprot_acc}-F1-model_v*.pdb"))
    if existing:
        return existing[0]

    label = name or uniprot_acc
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_acc}"
    try:
        entries = json.loads(http_get(api_url, timeout=20))
        if not entries:
            print(f"  {label}: not in AlphaFold DB")
            return None
        pdb_url = entries[0].get("pdbUrl", "")
        if not pdb_url:
            return None
        fname   = pdb_url.split("/")[-1]
        out     = output_dir / fname
        out.write_bytes(http_get(pdb_url, timeout=30))
        print(f"  {label}: downloaded → {out.name}")
        return out
    except Exception as e:
        print(f"  {label}: failed — {e}")
        return None


def download_alphafold_batch(
    targets:    dict[str, str],   # {name: uniprot_acc}
    output_dir: str | Path,
) -> dict[str, Optional[Path]]:
    """Download AlphaFold structures for multiple proteins."""
    return {name: download_alphafold(acc, output_dir, name=name)
            for name, acc in targets.items()}


# ── Centroid selection ────────────────────────────────────────────────────────

def select_centroid(
    cluster_label:  str,
    assignments_df,              # pd.DataFrame with protein_id + label column
    embeddings:     np.ndarray,
    embed_ids:      list[str],
    source_fastas:  list[str | Path],
    label_col:      str = "label",
) -> Optional[tuple[str, str]]:
    """
    Find the sequence nearest the embedding centroid for a given cluster label.
    Returns (protein_id, sequence) or None.
    """
    from .utils import read_fasta
    members = assignments_df.loc[assignments_df[label_col] == cluster_label, "protein_id"].tolist()
    if not members:
        return None

    id_to_idx = {pid: i for i, pid in enumerate(embed_ids)}
    valid     = [pid for pid in members if pid in id_to_idx]
    if not valid:
        return None

    sub_emb  = embeddings[[id_to_idx[pid] for pid in valid]]
    centroid = sub_emb.mean(axis=0)
    best_pid = valid[int(np.linalg.norm(sub_emb - centroid, axis=1).argmin())]

    for faa in source_fastas:
        for rid, seq in read_fasta(faa):
            if rid == best_pid:
                return best_pid, seq
    return None


# ── ESMFold ───────────────────────────────────────────────────────────────────

_esmfold_model    = None
_esmfold_tokenizer = None

def _load_esmfold():
    global _esmfold_model, _esmfold_tokenizer
    if _esmfold_model is None:
        from transformers import EsmForProteinFolding, AutoTokenizer
        print("  Loading ESMFold model (~2.8 GB, once per session) ...")
        _esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        _esmfold_model     = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        ).eval().to("cpu")
        print("  ESMFold ready.")
    return _esmfold_model, _esmfold_tokenizer


def _outputs_to_pdb(outputs) -> str:
    import torch
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats   import atom14_to_atom37

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    out_np = {k: v.to("cpu").numpy() for k, v in outputs.items()
              if isinstance(v, torch.Tensor)}
    pos_np = final_atom_positions.cpu().numpy()
    mask   = out_np["atom37_atom_exists"]
    i = 0
    return to_pdb(OFProtein(
        aatype         = out_np["aatype"][i],
        atom_positions = pos_np[i],
        atom_mask      = mask[i],
        residue_index  = out_np["residue_index"][i] + 1,
        b_factors      = out_np["plddt"][i],
        chain_index    = out_np["chain_index"][i] if "chain_index" in out_np else None,
    ))


def fold_esmfold(
    sequence:   str,
    label:      str,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Fold a sequence with ESMFold (local model).
    Returns path to PDB or None on failure.
    """
    import torch
    out = Path(output_dir) / f"{label}.pdb"
    if out.exists():
        return out
    print(f"  {label}: folding ({len(sequence)} aa) ...")
    model, tokenizer = _load_esmfold()
    try:
        tokens = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**tokens)
        out.write_text(_outputs_to_pdb(outputs))
        print(f"  {label}: saved → {out.name}")
        return out
    except Exception as e:
        print(f"  {label}: failed — {e}")
        return None
