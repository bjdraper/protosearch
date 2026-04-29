# Plan: NVIDIA NIM ESM-2 Embedding Backend

## Context

The local ESM-2 650M model (`esm2_t33_650M_UR50D`) is ~2.5 GB and takes 5–10 minutes
to download to Colab on every session. NVIDIA hosts the same model as a free REST API
via their NIM platform (api.nvidia.com) — no download, no GPU required. This plan adds
a `backend` parameter to the embedding layer so users can choose between local ESM-2
and the NVIDIA NIM API. The two backends produce identical output (1280-dim float32
mean-pool over layer 33), so downstream clustering/tree code is unchanged.

## API Contract (NVIDIA NIM)

- **Endpoint:** `https://api.nvidia.com/v1/embeddings`
- **Auth:** `Authorization: Bearer nvapi-...`
- **Request:**
  ```json
  { "input": ["MKTVRQ...", "ACDEFG..."], "model": "meta/esm2-650m" }
  ```
- **Response:**
  ```json
  { "data": [{"index": 0, "embedding": [0.012, ...]}, ...] }
  ```
  Each embedding is 1280 floats, mean-pooled from the final layer — same as local.
- **Rate limit:** 40 requests/min (free tier). Plan for ~10 sequences/request,
  with 1.6 s sleep between batches to stay under limit.
- **API key:** Free at https://build.nvidia.com/settings/api-keys (prefix `nvapi-`)

## Files to modify

| File | Change |
|------|--------|
| `protosearch/protosearch/embed.py` | Add `_embed_nvidia()` + `backend` / `api_key` params to `embed_sequences` and `embed_fasta` |
| `protosearch/notebooks/protein_family_survey.ipynb` | Insert cell `02b-apikey`; update cells `11-embed-hits` and `12-embed-ref` |
| `protosearch/requirements.txt` | Add `requests` |

## embed.py changes

### 1. New private function `_embed_nvidia()`

Insert before `embed_sequences`:

```python
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
```

### 2. Updated `embed_sequences()` — add `backend` and `api_key` params

```python
def embed_sequences(
    sequences:  list[tuple[str, str]],
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 32,
    device:     str = "cuda",
    layer:      int = 33,
    backend:    str = "local",   # "local" | "nvidia"
    api_key:    str = "",        # nvapi-... key; falls back to NVIDIA_API_KEY env var
) -> tuple[np.ndarray, list[str]]:
    if backend == "nvidia":
        import os
        key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            raise ValueError("backend='nvidia' requires api_key or NVIDIA_API_KEY env var")
        return _embed_nvidia(sequences, key, batch_size=min(batch_size, 10))
    # ... rest of existing local implementation unchanged ...
```

### 3. Updated `embed_fasta()` — pass `backend` and `api_key` through to `embed_sequences`

Add `backend: str = "local"` and `api_key: str = ""` to the signature and forward them.

## Notebook changes

### New cell `02b-apikey` (insert after `02-install`, before `03-drive`)

Cell id: `02b-apikey`, type: code

```python
# [02b] Embedding backend  ── USER INPUT ──────────────────────────────────────
# Option A — NVIDIA NIM (recommended for Colab: no 2.5 GB download, no GPU needed):
#   1. Get a free key at https://build.nvidia.com/settings/api-keys
#   2. Paste it below — EMBED_BACKEND is set automatically.
#
# Option B — Local ESM2 (needs Colab GPU runtime, downloads ~2.5 GB on first run):
#   Leave NVIDIA_API_KEY = '' — EMBED_BACKEND falls back to 'local'.

NVIDIA_API_KEY = ''           # paste nvapi-... key here
EMBED_BACKEND  = 'nvidia' if NVIDIA_API_KEY else 'local'
print(f'Embedding backend: {EMBED_BACKEND}')
# ─────────────────────────────────────────────────────────────────────────────
```

Use `NotebookEdit` with `edit_mode="insert"` and `cell_id="02-install"` to place it after the install cell.

### Updated cell `11-embed-hits`

```python
# [11] ESM2 embedding of HMMER hits
from protosearch import embed
hits_emb_path = DATA_DIR / 'embeddings' / 'hmmer_hits.npy'
hits_ids_path = DATA_DIR / 'embeddings' / 'hmmer_hits_ids.txt'
embed.embed_fasta(hmmer_hits_fasta, hits_emb_path, hits_ids_path,
                  backend=EMBED_BACKEND, api_key=NVIDIA_API_KEY)
print('Embeddings saved.')
```

### Updated cell `12-embed-ref`

```python
# [12] ESM2 embedding of reference probes
ref_emb_path = DATA_DIR / 'embeddings' / 'ref_embeddings.npy'
ref_ids_path = DATA_DIR / 'embeddings' / 'ref_embeddings_ids.txt'
embed.embed_fasta(probe_fasta, ref_emb_path, ref_ids_path,
                  backend=EMBED_BACKEND, api_key=NVIDIA_API_KEY)
print('Reference embeddings saved.')
```

## requirements.txt

Add one line: `requests`

## What stays the same

- All downstream code (`cluster`, `tree`, `asr`, `visualize`) is untouched — both
  backends return `(np.ndarray [N, 1280] float32, list[str])`.
- The full local path (`load_model`, `embed_sequences` body, `save_embeddings`) is
  completely unchanged; `backend='local'` is the default.
- `fair-esm` and `torch` stay in requirements for local-backend users.

## Verification

1. Get a free NVIDIA API key from `build.nvidia.com/settings/api-keys`
2. In Colab: paste key in cell `02b`, run cells 11 and 12
3. Assert `np.load(hits_emb_path).shape == (N, 1280)` and dtype is float32
4. Run cells 16–18 (cluster + t-SNE) — output should be identical to local run
5. Smoke-test fallback: set `NVIDIA_API_KEY = ''` → `EMBED_BACKEND = 'local'` → old behaviour
