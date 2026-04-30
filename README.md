# protosearch

Generic protein family survey pipeline. Accepts any Pfam domain ID or seed FASTA and surveys a user-supplied proteome.

**Pipeline:** HMMER prefilter → ESM2 embedding → Leiden clustering → t-SNE visualisation → IQ-TREE2 phylogeny → Ancestral state reconstruction → Cluster labeling

---

## Quick start (Google Colab)

Open `notebooks/protein_family_survey.ipynb` in Colab and fill in the three `USER INPUT` cells:

| Cell | What to fill in |
|------|-----------------|
| `[02b]` | NVIDIA NIM API key (or leave blank to use local ESM2) |
| `[04]` | Reference probe UniProt accessions + Pfam domain ID(s) |
| `[03]` | Survey name (must match your Google Drive folder) |

Run top to bottom. No GPU required if you use the NVIDIA NIM backend.

---

## Embedding backends

The pipeline supports two backends for ESM2-650M embeddings, selectable in cell `[02b]`:

### Option A — NVIDIA NIM (recommended for Colab)

- No model download, no GPU runtime needed
- Get a free API key at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
- Paste the `nvapi-...` key into cell `[02b]`
- Endpoint: `health.api.nvidia.com/v1/biology/meta/esm2-650m`
- Free tier: 40 requests/min; the pipeline throttles automatically

### Option B — Local ESM2

- Downloads the ESM2-650M model (~2.5 GB) on first run
- Requires a Colab GPU runtime (`Runtime > Change runtime type > T4 GPU`)
- Leave `NVIDIA_API_KEY = ''` in cell `[02b]` — the pipeline falls back automatically

Both backends produce identical output: 1280-dimensional float32 mean-pooled embeddings. All downstream clustering and tree code is unaffected by the choice of backend.

---

## Repository layout

```
protosearch/
├── protosearch/          ← Python package (8 modules)
│   ├── config.py         ← config loader
│   ├── utils.py          ← FASTA I/O, UniProt fetch, length filter, dedup
│   ├── search.py         ← HMMER runner, FAISS index/query, HMM download
│   ├── embed.py          ← ESM2 embedding (local + NVIDIA NIM backends)
│   ├── cluster.py        ← PCA, Leiden clustering, t-SNE
│   ├── tree.py           ← MAFFT, FastTree, IQ-TREE2
│   ├── asr.py            ← IQ-TREE2 .state parser, consensus sequences
│   └── visualize.py      ← t-SNE plots, sequence logos, ancestral tables
├── notebooks/
│   ├── protein_family_survey.ipynb   ← generic Colab template
│   └── agpat_crustacea.ipynb         ← AGPAT worked example
├── examples/
│   └── acyltransferase/config_example.yaml
├── requirements.txt
├── environment.colab.yml
└── setup.py
```

---

## Installation (local development)

```bash
conda create -n protosearch python=3.11
conda activate protosearch
pip install -e .
pip install fair-esm faiss-cpu leidenalg python-igraph openTSNE
```

HMMER, MAFFT, and IQ-TREE2 must be installed separately (available via `apt` on Linux/Colab; via `conda-forge` on macOS).

---

## Package usage

```python
from protosearch import embed, search, cluster, tree, asr, visualize
from protosearch import load_config, Config

# Embed with NVIDIA NIM
embeddings, ids = embed.embed_fasta(
    "hits.faa", "hits.npy", "hits_ids.txt",
    backend="nvidia", api_key="nvapi-..."
)

# Embed locally (default)
embeddings, ids = embed.embed_fasta("hits.faa", "hits.npy", "hits_ids.txt")
```

---

## Changelog

### NVIDIA NIM embedding backend

- Added `_embed_nvidia()` to `embed.py`: batched POST requests to the NVIDIA NIM biology API, returning mean-pooled 1280-dim ESM2-650M embeddings as a binary npz response
- `embed_sequences()` and `embed_fasta()` now accept `backend="nvidia"` and `api_key` parameters; `backend="local"` remains the default and the local code path is unchanged
- New cell `[02b]` in `protein_family_survey.ipynb`: user pastes API key; `EMBED_BACKEND` auto-selects `nvidia` or falls back to `local`
- Cells `[11]` and `[12]` updated to pass `backend` and `api_key` through to `embed_fasta`
- Added `requests` to `requirements.txt`

---

## Limitations

- NVIDIA NIM free tier caps sequences at 1024 residues each and 32 per request
- Local ESM2 backend requires ~8 GB GPU VRAM for large batches; reduce `batch_size` if you hit OOM
- IQ-TREE2 ASR is memory-intensive for alignments >500 sequences; split by cluster first (cells `[20–21]` do this automatically)
