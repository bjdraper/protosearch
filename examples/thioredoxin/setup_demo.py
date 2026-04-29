#!/usr/bin/env python3
"""
setup_demo.py — Download thioredoxin demo sequences from UniProt.

Run from this directory:
    python setup_demo.py

Creates: thioredoxin_demo/data/proteins_raw/demo_proteome.faa (~40 sequences)
Then upload the thioredoxin_demo/ folder to your Google Drive root (MyDrive/).
"""
import sys
import urllib.request
import urllib.parse
from pathlib import Path

OUT_DIR   = Path("thioredoxin_demo/data/proteins_raw")
OUT_FASTA = OUT_DIR / "demo_proteome.faa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://rest.uniprot.org/uniprotkb/search"

def fetch_fasta(query: str, size: int) -> str:
    params = urllib.parse.urlencode({
        "query":  query,
        "format": "fasta",
        "size":   size,
    })
    url = f"{BASE}?{params}"
    print(f"  → {url[:100]}")
    req = urllib.request.Request(url, headers={"User-Agent": "protosearch-demo/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

# ── Thioredoxin family sequences from diverse organisms ───────────────────────
print("Fetching thioredoxin family sequences...")
thx = fetch_fasta(
    "(reviewed:true) AND (protein_name:thioredoxin) "
    "AND (length:[80 TO 170]) "
    "NOT (protein_name:reductase) "
    "NOT (protein_name:peroxidase) "
    "NOT (protein_name:glutaredoxin)",
    size=30,
)

# ── Background proteins (non-thioredoxin, similar size range) ─────────────────
print("Fetching background sequences...")
bg = fetch_fasta(
    "(reviewed:true) "
    "AND (length:[80 TO 170]) "
    "NOT (protein_name:thioredoxin) "
    "NOT (protein_name:glutaredoxin) "
    "NOT (protein_name:oxidoreductase) "
    "NOT (protein_name:reductase) "
    "AND (organism_id:9606 OR organism_id:562 OR organism_id:559292)",
    size=12,
)

combined = thx.rstrip("\n") + "\n" + bg.rstrip("\n") + "\n"
n = combined.count(">")

OUT_FASTA.write_text(combined)
print(f"\nDone: {n} sequences → {OUT_FASTA}")
print("""
Next steps:
  1. Upload thioredoxin_demo/ to Google Drive root (MyDrive/thioredoxin_demo/)
  2. Open protosearch/notebooks/protein_family_survey.ipynb in Colab
  3. In cell [03] set:  SURVEY_NAME = 'thioredoxin_demo'
  4. In cell [04] set the values from config.yaml (see below)
  5. In cell [17] set:  resolution=0.5   (small dataset)
  6. In cell [18] set:  perplexity=8     (must be < N/3 where N = number of hits)
""")
