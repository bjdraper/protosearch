"""Shared utilities: FASTA I/O, HTTP retry, sequence deduplication."""

import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterator


# ── FASTA I/O ─────────────────────────────────────────────────────────────────

def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Return list of (id, sequence) from a FASTA file."""
    records, cur_id, cur_seq = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                records.append((cur_id, "".join(cur_seq)))
            cur_id = line[1:].split()[0]
            cur_seq = []
        elif line.strip():
            cur_seq.append(line.strip())
    if cur_id is not None:
        records.append((cur_id, "".join(cur_seq)))
    return records


def write_fasta(records: list[tuple[str, str]], path: str | Path) -> None:
    Path(path).write_text(
        "".join(f">{rid}\n{seq}\n" for rid, seq in records)
    )


def deduplicate_fasta(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove duplicate sequences (keep first occurrence by ID, then by sequence)."""
    seen_ids, seen_seqs, out = set(), set(), []
    for rid, seq in records:
        if rid not in seen_ids and seq not in seen_seqs:
            seen_ids.add(rid)
            seen_seqs.add(seq)
            out.append((rid, seq))
    return out


def filter_by_length(records: list[tuple[str, str]],
                     min_len: int = 200,
                     max_len: int = 500) -> list[tuple[str, str]]:
    return [(rid, seq) for rid, seq in records if min_len <= len(seq) <= max_len]


def combine_fastas(*paths: str | Path,
                   deduplicate: bool = True) -> list[tuple[str, str]]:
    records = []
    for p in paths:
        if Path(p).exists():
            records.extend(read_fasta(p))
    return deduplicate_fasta(records) if deduplicate else records


# ── HTTP utilities ────────────────────────────────────────────────────────────

def http_get(url: str, timeout: int = 30, retries: int = 5) -> bytes:
    """GET with exponential backoff retry."""
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def fetch_uniprot_fasta(accession: str) -> str:
    """Fetch a single UniProt sequence as FASTA string."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    return http_get(url, timeout=20).decode()


def fetch_uniprot_sequence(accession: str) -> tuple[str, str]:
    """Return (protein_id, sequence) for a UniProt accession."""
    fasta = fetch_uniprot_fasta(accession)
    lines = fasta.strip().splitlines()
    seq_id = lines[0][1:].split()[0]
    seq    = "".join(lines[1:])
    return seq_id, seq


def fetch_uniprot_sequences(accessions: list[str], output_path: str | Path) -> None:
    """Fetch multiple UniProt accessions and write them to a FASTA file."""
    records = [fetch_uniprot_sequence(acc) for acc in accessions]
    write_fasta(records, output_path)


def filter_and_dedup(
    in_fasta:  str | Path,
    out_fasta: str | Path,
    min_len:   int = 200,
    max_len:   int = 500,
) -> int:
    """Read FASTA, filter by length, deduplicate, write. Returns sequence count."""
    records = read_fasta(in_fasta)
    records = filter_by_length(records, min_len, max_len)
    records = deduplicate_fasta(records)
    write_fasta(records, out_fasta)
    return len(records)


def extract_sequences(
    fasta_path:  str | Path,
    ids:         list[str],
    output_path: str | Path,
) -> int:
    """Extract sequences matching ids from fasta_path and write to output_path."""
    id_set  = set(ids)
    records = [(rid, seq) for rid, seq in read_fasta(fasta_path) if rid in id_set]
    write_fasta(records, output_path)
    return len(records)
