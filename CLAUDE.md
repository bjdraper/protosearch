# CLAUDE.md — protosearch

Machine-readable project summary for Claude Code and future sessions.

## Project Overview

**protosearch** is a generalised protein family survey pipeline.  
Current application: discovering AGPAT-like (acyltransferase) proteins across 172 crustacean species.

The pipeline takes a set of reference protein sequences (UniProt accessions), a proteome FASTA, and a Pfam domain ID, then:
1. Filters and deduplicates proteins by length
2. Runs HMMER domain prefilter (PF01553 for acyltransferases)
3. Embeds hits with ESM2-650M (Meta protein language model)
4. Clusters embeddings with Leiden community detection + t-SNE
5. Queries a FAISS KNN index with reference probes to find nearest neighbours
6. Builds phylogenetic trees per cluster (MAFFT + FastTree)
7. Runs ancestral state reconstruction with IQ-TREE2

## GitHub Repository

- **Repo:** https://github.com/bjdraper/protosearch
- **Visibility:** Public
- **Local path:** `/Users/bendraper/project02/crustacea/protosearch/`
- **Branch:** main
- **Sync method:** GitHub Desktop → commit + push locally, pull/clone in Colab

## Analysis Configuration (AGPAT Example)

Active config: `config.yaml` (copied from `config_example_acyltransferase.yaml`)

| Parameter | Value |
|---|---|
| Pfam domain | PF01553 (1-acylglycerol-3-phosphate acyltransferase) |
| Length filter | 200–500 AA |
| ESM2 model | esm2_t33_650M_UR50D (650M, layer 33, dim 1280) |
| FAISS index | IndexFlatL2 |
| Leiden resolution | 2.0 (top-level), 1.0 (sub-clustering) |
| Tree model | FastTree LG; IQ-TREE2 LG+G4 |

**Reference probes (20 total):** AGPAT1–5 (human), LCLAT1 (human/mouse/zebrafish), GPAT3/4 (human), GPAT1/2 (mouse), AGPAT2 (mouse), LPCAT2 (human), PlsC (E.coli/B.sub/diatom), Tafazzin and LPCAT3/MBOAT7 as outgroup controls.

**Catalytic motifs highlighted:** HxxD (catalytic dyad), FPxG (acyl-CoA binding), EGTR (Block II conserved).

## Data Storage

Data is NOT committed to GitHub (too large). Two locations:

### Local (macOS, Ben's machine)
```
/Users/bendraper/project02/crustacea/
  data/
    crustome/              # raw proteome FASTAs — 172 crustacean species
    crustome_filtered/     # length-filtered FASTAs
    hmm_profiles/          # downloaded Pfam HMM
    hmmer_hits/            # HMMER-filtered sequences
    embeddings/            # ESM2 .npy arrays + id lists
    knn_index/             # FAISS index + id_map.tsv
    query_sequences/       # reference probe FASTAs
  results/                 # TSVs, plots, trees, alignments
  config/
    species_list.yaml      # 172 crustacean species with NCBI accessions
```

### Google Colab (Google Drive mount)
```
/content/drive/MyDrive/agpat_crustacea/
  data/                    # same structure as above, populated at runtime
  results/
  config.yaml              # copied from repo at setup
```

## Crustacea Genome Dataset

The proteome used is a **custom set of 172 crustacean species** assembled from NCBI:
- 27 RefSeq + 145 GenBank assemblies
- Groups: Malacostraca (116), Branchiopoda (31), Copepoda (22), other (3)
- Assembly levels: Chromosome (59), Scaffold (89), Contig (20), Complete (4)
- Species list with NCBI accessions: `crustacea/config/species_list.yaml`

**This dataset is not publicly distributed with the repo.** A new user must either:
1. Upload their own proteome FASTA to Google Drive and set `INPUT_FASTAS` in notebook cell [08]
2. Re-fetch the crustacean genomes from NCBI using `species_list.yaml` and the original pipeline scripts in `/Users/bendraper/project02/scripts/`

## Google Colab Notebook

**Single notebook:** `notebooks/agpat_crustacea.ipynb`  
**Runtime:** T4 GPU recommended (Runtime → Change runtime type → T4 GPU)

### Cell flow
| Cell | What it does |
|---|---|
| 00 | Clone repo from GitHub + add to sys.path |
| 02 | Install system tools (hmmer, mafft, fasttree) + pip packages |
| 03 | Mount Google Drive; **user sets `SURVEY_NAME`** → drives `PROJECT_ROOT` |
| 04 | Define protein family — pre-filled with AGPAT probes; user replaces for other families |
| 05 | Write config.yaml |
| 06 | Download reference probes from UniProt |
| 07 | Download Pfam HMM profile |
| 08 | **Drive FASTA scanner** — lists `.faa`/`.fasta` files on Drive; **user sets `INPUT_FASTAS`**; combines to `raw_fasta` |
| 09 | Filter proteins by length + dedup |
| 10 | HMMER prefilter |
| 11 | ESM2 embedding of hits |
| 12 | ESM2 embedding of reference probes |
| 13 | Build FAISS KNN index |
| 14 | Query KNN with reference probes |
| 15 | Distance distribution plot |
| 16 | Load embeddings for clustering |
| 17 | Leiden clustering |
| 18 | t-SNE plot |
| 19 | Sub-clustering (optional) |
| 20 | MAFFT + FastTree per cluster |
| 21 | IQ-TREE2 ASR |
| 22 | Parse ASR, generate sequence logos + ancestral tables |

### To use with a different proteome
1. In cell [03], set `SURVEY_NAME` (e.g. `'kinase_insects'`) — this becomes your Drive subfolder
2. In cell [04], replace `REFERENCE_PROBES` and `PFAM_IDS` for your protein family
3. Upload your `.faa` / `.fasta` file(s) to `MyDrive/{SURVEY_NAME}/data/proteins_raw/`
4. Run cell [08] — it scans Drive and lists found files; set `INPUT_FASTAS` to match
5. Run remaining cells top to bottom

### Dependencies installed at runtime
```
# System (apt-get)
hmmer=3.4, mafft=7.526, fasttree=2.2.0

# Python (pip)
numpy, pandas, scipy, scikit-learn, biopython,
fair-esm, faiss-cpu, leidenalg, python-igraph, openTSNE,
ete3, logomaker, py3Dmol, transformers, accelerate,
seaborn, pyyaml, tqdm
```

IQ-TREE2 is downloaded from GitHub releases at runtime (not via pip/apt).

## Python Package

The `acyltransferase/` directory is the project's internal Python package:

| Module | Purpose |
|---|---|
| `config.py` | Load and validate config.yaml |
| `utils.py` | FASTA I/O, UniProt fetch, length filter, dedup |
| `search.py` | HMMER runner, FAISS index build + query, HMM download |
| `embed.py` | ESM2 embedding via fair-esm, save/load .npy |
| `cluster.py` | PCA, FAISS KNN graph, Leiden, t-SNE |
| `tree.py` | MAFFT alignment, FastTree, IQ-TREE2 runner |
| `asr.py` | Parse IQ-TREE2 .state, consensus sequences, variable positions |
| `visualize.py` | t-SNE plots, sequence logos, ancestral tables |
| `structure.py` | AlphaFold download, ESMFold prediction, centroid selection |
| `structure_viz.py` | py3Dmol viewers, RMSD table (not used in current notebook) |

## Environment (local dev)

```
conda env: agpat_tree (Python 3.11)
Platform: macOS Apple Silicon (arm64)
Key versions: numpy=2.4.3, pandas=3.0.2, scikit-learn=1.8.0,
              biopython=1.87, fair-esm=2.0.0, faiss-cpu=1.13.2,
              torch=2.11.0, transformers=5.6.2, leidenalg=0.11.0
```

## Aims of the Analysis

1. **Discovery:** Identify AGPAT-like acyltransferase proteins across Crustacea using embedding-based similarity rather than sequence identity alone
2. **Phylogenetics:** Reconstruct evolutionary relationships of crustacean acyltransferases relative to well-characterised human/bacterial homologs
3. **Ancestral reconstruction:** Infer ancestral sequences at key nodes (e.g. LCLAT1, GPAT4 crustacean clades) to understand functional divergence
4. **Generalisation:** The pipeline is protein-family agnostic — swap `config.yaml` to survey any domain family across any set of proteomes
