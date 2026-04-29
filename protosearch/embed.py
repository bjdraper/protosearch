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


def embed_sequences(
    sequences: list[tuple[str, str]],   # [(id, seq), ...]
    model_name: str  = "esm2_t33_650M_UR50D",
    batch_size: int  = 32,
    device: str      = "cuda",
    layer: int       = 33,
) -> tuple[np.ndarray, list[str]]:
    """
    Embed protein sequences with ESM2.
    Returns (embeddings float32 array [N, 1280], list of ids in same order).
    """
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
