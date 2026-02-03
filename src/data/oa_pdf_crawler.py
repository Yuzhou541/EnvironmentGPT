from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

OPENALEX_API = "https://api.openalex.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2"


def _safe_filename(s: str, max_len: int = 160) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s[:max_len] if len(s) > max_len else s


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _is_pdf_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return "application/pdf" in ctype or resp.url.lower().endswith(".pdf")


def _download_pdf(url: str, out_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    """
    Download a PDF from URL to out_path. Returns (ok, message).
    This function only downloads if the response looks like a PDF.
    """
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            if not _is_pdf_response(r):
                return False, f"Not a PDF (Content-Type={r.headers.get('Content-Type','')})"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")

            size = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)

            # Basic sanity check: avoid saving tiny HTML error pages
            if size < 20_000:
                tmp_path.unlink(missing_ok=True)
                return False, f"File too small ({size} bytes)"

            tmp_path.replace(out_path)
            return True, "OK"
    except requests.exceptions.RequestException as e:
        return False, f"RequestException: {e}"


@dataclass
class CrawlResult:
    source: str
    work_id: str
    title: str
    doi: str
    pdf_url: str
    landing_url: str
    saved_path: str
    status: str
    note: str


def _write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _openalex_search(query: str, max_results: int, polite_delay: float) -> Iterable[Dict[str, Any]]:
    """
    Yield OpenAlex works that are likely OA and have some OA location.
    """
    per_page = 25
    cursor = "*"
    fetched = 0

    while fetched < max_results:
        params = {
            "search": query,
            "per-page": per_page,
            "cursor": cursor,
            # Request a bit more fields by default
        }
        r = requests.get(OPENALEX_API, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAlex error: HTTP {r.status_code} {r.text[:200]}")

        data = r.json()
        results = data.get("results", [])
        cursor = data.get("meta", {}).get("next_cursor", None)
        if not results:
            break

        for w in results:
            yield w
            fetched += 1
            if fetched >= max_results:
                break

        if cursor is None:
            break
        time.sleep(polite_delay)


def _pick_best_pdf_url_from_openalex(work: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (pdf_url, landing_url) where pdf_url may be empty.
    """
    landing_url = ""
    # Prefer OpenAlex best_oa_location
    best = work.get("best_oa_location") or {}
    landing_url = best.get("landing_page_url") or ""
    pdf_url = best.get("pdf_url") or ""

    # Fallback: scan oa_locations
    if not pdf_url:
        for loc in (work.get("oa_locations") or []):
            if loc.get("pdf_url"):
                pdf_url = loc["pdf_url"]
                landing_url = loc.get("landing_page_url") or landing_url
                break

    return pdf_url or "", landing_url or ""


def _normalize_doi(doi: str) -> str:
    doi = (doi or "").strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return doi.lower()


def _unpaywall_lookup(doi: str, email: str, polite_delay: float) -> Dict[str, Any]:
    url = f"{UNPAYWALL_API}/{doi}"
    params = {"email": email}
    r = requests.get(url, params=params, timeout=60)
    time.sleep(polite_delay)
    if r.status_code != 200:
        raise RuntimeError(f"Unpaywall error: DOI={doi} HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def _pick_best_pdf_url_from_unpaywall(obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (pdf_url, landing_url) where pdf_url may be empty.
    """
    best = obj.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf") or ""
    landing_url = best.get("url") or ""

    # Fallback scan oa_locations
    if not pdf_url:
        for loc in (obj.get("oa_locations") or []):
            if loc.get("url_for_pdf"):
                pdf_url = loc["url_for_pdf"]
                landing_url = loc.get("url") or landing_url
                break
    return pdf_url or "", landing_url or ""


def _read_dois(path: Path) -> List[str]:
    dois = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = _normalize_doi(line)
            if d:
                dois.append(d)
    return dois


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["openalex", "unpaywall"], required=True)

    ap.add_argument("--query", type=str, default="", help="OpenAlex search query (required for mode=openalex).")
    ap.add_argument("--max_results", type=int, default=200, help="Max works to consider for OpenAlex mode.")

    ap.add_argument("--doi_file", type=str, default="", help="Path to a text file with one DOI per line (mode=unpaywall).")

    ap.add_argument("--out_dir", type=str, required=True, help="Directory to save PDFs.")
    ap.add_argument("--meta_out", type=str, required=True, help="JSONL metadata output path.")
    ap.add_argument("--polite_delay", type=float, default=0.8, help="Sleep seconds between requests.")
    ap.add_argument("--timeout", type=int, default=60)

    # Windows + conda run may split a quoted query into multiple argv tokens.
    # We parse known args and then stitch unknown tokens back into the query.
    args, unknown = ap.parse_known_args()

    if args.mode == "openalex":
        # If query got split, argparse will parse only the first token into args.query
        # and the rest appear in `unknown`. Stitch them back.
        if unknown:
            if args.query:
                args.query = (str(args.query) + " " + " ".join(unknown)).strip()
            else:
                args.query = " ".join(unknown).strip()


    out_dir = Path(args.out_dir)
    meta_out = Path(args.meta_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_keys = set()
    saved = 0
    skipped = 0

    if args.mode == "openalex":
        if not args.query.strip():
            raise ValueError("--query is required for mode=openalex")

        works = list(_openalex_search(args.query, args.max_results, args.polite_delay))
        pbar = tqdm(works, desc="OpenAlex works")
        for w in pbar:
            title = (w.get("title") or "").strip()
            doi = _normalize_doi(w.get("doi") or "")
            work_id = w.get("id") or ""
            pdf_url, landing_url = _pick_best_pdf_url_from_openalex(w)

            key = doi or work_id or title
            key_hash = _sha1(key)
            if key_hash in seen_keys:
                skipped += 1
                continue
            seen_keys.add(key_hash)

            if not pdf_url:
                _write_jsonl(meta_out, {
                    "source": "openalex",
                    "work_id": work_id,
                    "title": title,
                    "doi": doi,
                    "pdf_url": "",
                    "landing_url": landing_url,
                    "saved_path": "",
                    "status": "NO_PDF_URL",
                    "note": "No OA PDF url found in OpenAlex record",
                })
                skipped += 1
                continue

            fn_base = _safe_filename(doi if doi else (title[:80] if title else work_id))
            pdf_path = out_dir / f"{fn_base}_{key_hash[:10]}.pdf"

            if pdf_path.exists():
                saved += 1
                continue

            ok, note = _download_pdf(pdf_url, pdf_path, timeout=args.timeout)
            status = "DOWNLOADED" if ok else "FAILED"

            _write_jsonl(meta_out, {
                "source": "openalex",
                "work_id": work_id,
                "title": title,
                "doi": doi,
                "pdf_url": pdf_url,
                "landing_url": landing_url,
                "saved_path": str(pdf_path) if ok else "",
                "status": status,
                "note": note,
            })
            if ok:
                saved += 1
            else:
                skipped += 1

            pbar.set_postfix({"saved": saved, "skipped": skipped})

    elif args.mode == "unpaywall":
        doi_file = Path(args.doi_file)
        if not doi_file.exists():
            raise ValueError("--doi_file must exist for mode=unpaywall")

        email = os.environ.get("UNPAYWALL_EMAIL", "").strip()
        if not email:
            raise ValueError("Set environment variable UNPAYWALL_EMAIL for Unpaywall, e.g. $env:UNPAYWALL_EMAIL='you@example.com'")

        dois = _read_dois(doi_file)
        pbar = tqdm(dois, desc="Unpaywall DOIs")
        for doi in pbar:
            key_hash = _sha1(doi)
            if key_hash in seen_keys:
                skipped += 1
                continue
            seen_keys.add(key_hash)

            try:
                obj = _unpaywall_lookup(doi, email=email, polite_delay=args.polite_delay)
            except Exception as e:
                _write_jsonl(meta_out, {
                    "source": "unpaywall",
                    "work_id": "",
                    "title": "",
                    "doi": doi,
                    "pdf_url": "",
                    "landing_url": "",
                    "saved_path": "",
                    "status": "FAILED_LOOKUP",
                    "note": str(e),
                })
                skipped += 1
                continue

            title = (obj.get("title") or "").strip()
            pdf_url, landing_url = _pick_best_pdf_url_from_unpaywall(obj)

            if not pdf_url:
                _write_jsonl(meta_out, {
                    "source": "unpaywall",
                    "work_id": "",
                    "title": title,
                    "doi": doi,
                    "pdf_url": "",
                    "landing_url": landing_url,
                    "saved_path": "",
                    "status": "NO_PDF_URL",
                    "note": "No OA PDF url found via Unpaywall",
                })
                skipped += 1
                continue

            fn_base = _safe_filename(doi)
            pdf_path = out_dir / f"{fn_base}_{key_hash[:10]}.pdf"

            if pdf_path.exists():
                saved += 1
                continue

            ok, note = _download_pdf(pdf_url, pdf_path, timeout=args.timeout)
            status = "DOWNLOADED" if ok else "FAILED"

            _write_jsonl(meta_out, {
                "source": "unpaywall",
                "work_id": "",
                "title": title,
                "doi": doi,
                "pdf_url": pdf_url,
                "landing_url": landing_url,
                "saved_path": str(pdf_path) if ok else "",
                "status": status,
                "note": note,
            })

            if ok:
                saved += 1
            else:
                skipped += 1

            pbar.set_postfix({"saved": saved, "skipped": skipped})

    print(f"Done. Saved={saved}, skipped/failed={skipped}.")
    print(f"PDF dir: {out_dir}")
    print(f"Metadata JSONL: {meta_out}")


if __name__ == "__main__":
    main()
