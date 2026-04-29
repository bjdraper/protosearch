"""ESM2-650M protein embedding."""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional


def get_device(requested: str = "cuda") -> str:
    import torch
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str = "esm2_t33_650M_UR50D",
               device: Optional[str] = None):
    """Load ESM2 model + batch converter. Returns (model, alphabet, batch_converter)."""
    import torch, esm
    device = device or get_device()
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    return model, alphabet, alphabet.get_batch_converter(), device


def _embed_nvidia(
    sequences:  list[tuple[str, str]],
    api_key:    str,
    batch_size: int = 10,
    rpm_limit:  int = 40,
) -> tuple[np.ndarray, list[str]]:
    import requests, time

    url     = "https://api.nvidia.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    sleep_s = 60.0 / rpm_limit + 0.1   # ~1.6 s per batch to stay under 40 RPM

    all_emb, all_ids = [], []
    for i in range(0, len(sequences), batch_size):
        batch   = sequences[i : i + batch_size]
        payload = {"input": [seq for _, seq in batch], "model": "meta/esm2-650m"}
        resp    = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data    = resp.json()["data"]
        data.sort(key=lambda x: x["index"])           # API may return out of order
        all_emb.extend(np.array(d["embedding"], dtype=np.float32) for d in data)
        all_ids.extend(sid for sid, _ in batch)
        print(f"  embedded {min(i + batch_size, len(sequences))}/{len(sequences)}")
        if i + batch_size < len(sequences):
            time.sleep(sleep_s)

    return np.vstack(all_emb), all_ids


def embed_sequences(
    sequences:  list[tuple[str, str]],   # [(id, seq), ...]
    model_name: str  = "esm2_t33_650M_UR50D",
    batch_size: int  = 32,
    device:     str  = "cuda",
    layer:      int  = 33,
    backend:    str  = "local",   # "local" | "nvidia"
    api_key:    str  = "",        # nvapi-... key; falls back to NVIDIA_API_KEY env var
) -> tuple[np.ndarray, list[str]]:
    """
    Embed protein sequences with ESM2 (local) or NVIDIA NIM API.
    Returns (embeddings float32 array [N, 1280], list of ids in same order).
    """
    if backend == "nvidia":
        import os
        key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            raise ValueError("backend='nvidia' requires api_key or NVIDIA_API_KEY env var")
        return _embed_nvidia(sequences, key, batch_size=min(batch_size, 10))

    import torch
    device = get_device(device)
    model, alphabet, batch_converter, device = load_model(model_name, device)

    all_emb, all_ids = [], []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[layer], return_contacts=False)
        reps = out["representations"][layer]
        for j, (seq_id, seq) in enumerate(batch):
            # mean-pool over sequence positions (exclude BOS/EOS tokens)
            emb = reps[j, 1:len(seq) + 1].mean(0).cpu().numpy().astype(np.float32)
            all_emb.append(emb)
            all_ids.append(seq_id)
        if i % (batch_size * 10) == 0:
            print(f"  embedded {i + len(batch)}/{len(sequences)}")

    return np.vstack(all_emb), all_ids


def save_embeddings(embeddings: np.ndarray, ids: list[str],
                    npy_path: str | Path, ids_path: str | Path) -> None:
    np.save(npy_path, embeddings)
    Path(ids_path).write_text("\n".join(ids))


def embed_fasta(
    fasta_path: str | Path,
    emb_path:   str | Path,
    ids_path:   str | Path,
    device:     str = "cuda",
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 32,
    layer:      int = 33,
    backend:    str = "local",
    api_key:    str = "",
) -> tuple[np.ndarray, list[str]]:
    """Read a FASTA file, embed all sequences, and save embeddings + ids to disk."""
    from .utils import read_fasta
    Path(emb_path).parent.mkdir(parents=True, exist_ok=True)
    sequences  = read_fasta(fasta_path)
    embeddings, ids = embed_sequences(
        sequences, model_name=model_name, batch_size=batch_size,
        device=device, layer=layer, backend=backend, api_key=api_key,
    )
    save_embeddings(embeddings, ids, emb_path, ids_path)
    return embeddings, ids


def load_embeddings(*stems: tuple[Path, Path]) -> tuple[np.ndarray, list[str]]:
    """
    Load and concatenate multiple (npy_path, ids_path) pairs.
    stems: [(emb.npy, emb_ids.txt), ...]
    """
    parts_emb, parts_ids = [], []
    for npy, ids_txt in stems:
        if Path(npy).exists() and Path(ids_txt).exists():
            parts_emb.append(np.load(npy).astype(np.float32))
            parts_ids.extend(Path(ids_txt).read_text().splitlines())
    if not parts_emb:
        raise FileNotFoundError("No embedding files found")
    return np.vstack(parts_emb), parts_ids
