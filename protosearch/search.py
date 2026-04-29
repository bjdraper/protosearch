"""HMMER domain search + KNN query against FAISS index."""

from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ── HMMER ─────────────────────────────────────────────────────────────────────

def download_hmm_profile(pfam_id: str, output_path: str | Path) -> Path:
    """Download a Pfam HMM profile from EBI to a specific file path."""
    from .utils import http_get
    import gzip
    url  = f"https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfam_id}?annotation=hmm"
    data = http_get(url, timeout=60)
    try:
        data = gzip.decompress(data)
    except Exception:
        pass
    Path(output_path).write_bytes(data)
    return Path(output_path)


def download_pfam_hmm(pfam_id: str, hmm_dir: str | Path) -> Path:
    """Download a Pfam HMM profile into hmm_dir as <pfam_id>.hmm."""
    hmm_dir = Path(hmm_dir)
    hmm_dir.mkdir(parents=True, exist_ok=True)
    return download_hmm_profile(pfam_id, hmm_dir / f"{pfam_id}.hmm")


def run_hmmer(
    fasta_path:      str | Path,
    hmm_dir_or_path: str | Path,
    hits_fasta:      str | Path,
    hits_tsv:        str | Path | None = None,
    evalue:          float = 1e-5,
    cpu:             int   = 4,
    name_filter:     str   = "",
) -> Path:
    """
    Run hmmsearch and write hits to hits_fasta.

    hmm_dir_or_path: a single .hmm file OR a directory — all .hmm files in the
                     directory are searched and hits are merged.
    hits_tsv:        optional path for the domain table output.
    Returns path to hits_fasta.
    """
    from .utils import read_fasta, write_fasta, deduplicate_fasta

    hits_fasta = Path(hits_fasta)
    hits_fasta.parent.mkdir(parents=True, exist_ok=True)

    hmm_path = Path(hmm_dir_or_path)
    hmm_files = sorted(hmm_path.glob("*.hmm")) if hmm_path.is_dir() else [hmm_path]
    if not hmm_files:
        raise FileNotFoundError(f"No .hmm files found in {hmm_path}")

    all_hit_ids: set[str] = set()
    last_domtbl: Path | None = None

    for hmm_file in hmm_files:
        domtbl = hits_fasta.parent / f"{Path(fasta_path).stem}_{hmm_file.stem}.domtblout"
        subprocess.run(
            ["hmmsearch", "--domtblout", str(domtbl),
             "--noali", "--cpu", str(cpu), "-E", "0.01",
             str(hmm_file), str(fasta_path)],
            check=True, capture_output=True,
        )
        for line in domtbl.read_text().splitlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 14:
                continue
            if float(cols[12]) <= evalue:
                if not name_filter or name_filter.lower() in cols[3].lower():
                    all_hit_ids.add(cols[0])
        last_domtbl = domtbl

    if hits_tsv and last_domtbl:
        import shutil
        shutil.copy(last_domtbl, hits_tsv)

    all_records = read_fasta(fasta_path)
    hits = deduplicate_fasta([(rid, seq) for rid, seq in all_records if rid in all_hit_ids])
    write_fasta(hits, hits_fasta)
    return hits_fasta


# ── FAISS KNN index ──────────────────────────────────────────────────────────

def build_knn_index(embeddings: np.ndarray, ids: list[str],
                    index_path: str | Path, id_map_path: str | Path) -> None:
    """Build FAISS IndexFlatL2 and id_map TSV."""
    import faiss
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(index_path))
    pd.DataFrame({"row": range(len(ids)), "protein_id": ids}).to_csv(
        id_map_path, sep="\t", index=False
    )


def load_knn_index(index_path: str | Path, id_map_path: str | Path):
    """Return (faiss_index, id_map DataFrame)."""
    import faiss
    index  = faiss.read_index(str(index_path))
    id_map = pd.read_csv(id_map_path, sep="\t")
    return index, id_map


def query_knn(
    query_sequences: list[tuple[str, str]],
    index_path: str | Path,
    id_map_path: str | Path,
    k: int = 20,
    embed_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Embed query sequences, search FAISS index.
    Returns DataFrame with columns: query_id, rank, protein_id, l2_dist.
    """
    from .embed import embed_sequences
    embed_kwargs = embed_kwargs or {}
    q_emb, q_ids = embed_sequences(query_sequences, **embed_kwargs)

    index, id_map = load_knn_index(index_path, id_map_path)
    distances, indices = index.search(q_emb.astype(np.float32), k)

    rows = []
    for qi, qid in enumerate(q_ids):
        for rank, (dist, idx) in enumerate(zip(distances[qi], indices[qi])):
            hit_id = id_map.iloc[idx]["protein_id"]
            rows.append({"query_id": qid, "rank": rank + 1,
                         "protein_id": hit_id, "l2_dist": float(dist)})
    return pd.DataFrame(rows)
