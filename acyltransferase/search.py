"""HMMER domain search + KNN query against FAISS index."""

from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ── HMMER ─────────────────────────────────────────────────────────────────────

def download_hmm_profile(pfam_id: str, output_path: str | Path) -> Path:
    """Download a Pfam HMM profile from EBI."""
    from .utils import http_get
    url  = f"https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfam_id}?annotation=hmm"
    data = http_get(url, timeout=60)
    # EBI returns gzipped HMM
    import gzip, io
    try:
        data = gzip.decompress(data)
    except Exception:
        pass
    Path(output_path).write_bytes(data)
    return Path(output_path)


def run_hmmer(
    fasta_path: str | Path,
    hmm_path:   str | Path,
    output_dir: str | Path,
    evalue:     float = 1e-5,
    cpu:        int   = 4,
    name_filter: str  = "",
) -> Path:
    """
    Run hmmsearch against fasta_path. Return path to hits FASTA.
    name_filter: if set, only keep hits where query name contains this string.
    """
    from .utils import read_fasta, write_fasta
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem     = Path(fasta_path).stem
    domtbl   = output_dir / f"{stem}.domtblout"
    hits_faa = output_dir / f"{stem}_hmmer.faa"

    subprocess.run(
        ["hmmsearch", "--domtblout", str(domtbl),
         "--noali", f"--cpu", str(cpu), "-E", "0.01", str(hmm_path), str(fasta_path)],
        check=True, capture_output=True,
    )

    # parse domain table
    hit_ids = set()
    for line in domtbl.read_text().splitlines():
        if line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 14:
            continue
        query_name = cols[3]
        dom_ievalue = float(cols[12])
        if dom_ievalue <= evalue:
            if not name_filter or name_filter.lower() in query_name.lower():
                hit_ids.add(cols[0])  # target name (protein ID)

    all_records = read_fasta(fasta_path)
    hits = [(rid, seq) for rid, seq in all_records if rid in hit_ids]
    write_fasta(hits, hits_faa)
    return hits_faa


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
