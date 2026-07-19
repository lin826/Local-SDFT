#!/usr/bin/env python3
"""Scrape top PHD Comics from most_popular.php and build geek-jokes JSONL.

Caches HTML under data/phdcomics/raw/ for resume. Rate-limited and polite.

Training rows use setup/punchline jokes derived from comic titles — not the fake
journal abstracts. See ``scripts/phdcomics_joke_builder.py`` for the field convention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phdcomics_joke_builder import to_training_row  # noqa: E402

USER_AGENT = "Local-SDFT/1.0 (+https://github.com/local-sdft; research scraper)"
REQUEST_DELAY_S = 1.5
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5.0

DATA_DIR = ROOT / "data" / "phdcomics"
RAW_DIR = DATA_DIR / "raw"
PAGES_DIR = RAW_DIR / "pages"
JOURNALS_DIR = RAW_DIR / "journals"
MANIFEST_PATH = DATA_DIR / "manifest.json"
COMICS_JSONL = DATA_DIR / "comics.jsonl"
GEEK_JOKES_PATH = ROOT / "data" / "geek_jokes.jsonl"
MOST_POPULAR_URL = "https://phdcomics.com/comics/most_popular.php"


def _fetch(url: str, dest: Path | None = None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            text = body.decode("utf-8", errors="replace")
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8")
            return text
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}") from last_err


def _clean_text(text: str) -> str:
    text = unescape(re.sub(r"\s+", " ", text)).strip()
    return text.strip("'\"")


def parse_most_popular(html: str) -> list[dict]:
    """Extract ranked comics from the most-popular listing page."""
    rows: list[dict] = []
    pattern = re.compile(
        r"comicid=(\d+)><img[^>]*></td><td[^>]*><i>([^<]+)</i>",
        re.IGNORECASE,
    )
    popularity_pattern = re.compile(
        r"<td[^>]*>(\d+)</td>\s*<td[^>]*><a\s*\n?"
        r"href=[^>]*comicid=(\d+)><img",
        re.IGNORECASE | re.DOTALL,
    )
    popularity_by_id: dict[str, int] = {}
    for m in popularity_pattern.finditer(html):
        popularity_by_id[m.group(2)] = int(m.group(1))

    seen: set[str] = set()
    for rank, m in enumerate(pattern.finditer(html), start=1):
        comic_id = m.group(1)
        if comic_id in seen:
            continue
        seen.add(comic_id)
        rows.append(
            {
                "comic_id": comic_id,
                "rank": rank,
                "popularity": popularity_by_id.get(comic_id),
                "title": _clean_text(m.group(2)),
                "source": "phdcomics_most_popular",
            }
        )
    return rows


def parse_comic_page(html: str, comic_id: str) -> dict:
    title = ""
    m = re.search(r"<title>\s*PHD Comics:\s*(.+?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = _clean_text(m.group(1))

    image_url = ""
    for pat in (
        r'<meta property=[\'"]og:image[\'"] content=[\'"]([^\'"]+)[\'"]',
        r'id=comic2?\s+name=comic2?\s+src=([^\s>]+)',
    ):
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            image_url = m.group(1).strip("'\"")
            break

    image_alt = ""
    m = re.search(r'id=comic2?\s+[^>]*alt=[\'"]([^\'"]*)[\'"]', html, re.IGNORECASE)
    if m:
        image_alt = _clean_text(m.group(1))

    has_journal = "archive_journal.php" in html
    journal_id = comic_id if has_journal else ""

    return {
        "comic_id": comic_id,
        "title": title,
        "image_url": image_url,
        "image_alt": image_alt,
        "has_journal": has_journal,
        "journal_id": journal_id,
    }


def parse_journal_page(html: str) -> dict:
    title = ""
    m = re.search(r"<font size=\"\+2\">(.+?)</font>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = _clean_text(re.sub(r"<[^>]+>", "", m.group(1)))

    abstract = ""
    m = re.search(r"<b>Abstract</b>\s*<br>\s*<table[^>]*>\s*<tr>\s*<td>(.+?)</td>", html, re.IGNORECASE | re.DOTALL)
    if m:
        abstract = _clean_text(re.sub(r"<[^>]+>", " ", m.group(1)))

    return {"journal_title": title, "abstract": abstract}


def load_manifest() -> dict:
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"listing": [], "completed_ids": [], "source_url": MOST_POPULAR_URL}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def scrape_listing(*, force: bool) -> list[dict]:
    listing_path = RAW_DIR / "most_popular.html"
    if force or not listing_path.is_file():
        print(f"fetching listing {MOST_POPULAR_URL}")
        html = _fetch(MOST_POPULAR_URL, listing_path)
        time.sleep(REQUEST_DELAY_S)
    else:
        html = listing_path.read_text(encoding="utf-8")
    listing = parse_most_popular(html)
    if len(listing) < 200:
        raise RuntimeError(f"expected 200 comics, parsed {len(listing)}")
    return listing[:200]


def scrape_comics(
    listing: list[dict],
    *,
    limit: int | None,
    force: bool,
    fetch_journals: bool,
) -> list[dict]:
    manifest = load_manifest()
    manifest["listing"] = listing
    completed = set(manifest.get("completed_ids", []))
    records_by_id: dict[str, dict] = {}

    if COMICS_JSONL.is_file() and not force:
        for line in COMICS_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records_by_id[rec["comic_id"]] = rec

    targets = listing[: limit or len(listing)]
    for idx, item in enumerate(targets, start=1):
        comic_id = item["comic_id"]
        if comic_id in completed and comic_id in records_by_id and not force:
            continue

        page_path = PAGES_DIR / f"{comic_id}.html"
        page_url = f"https://phdcomics.com/comics/archive.php?comicid={comic_id}"
        if force or not page_path.is_file():
            print(f"[{idx}/{len(targets)}] comic {comic_id}: {item['title'][:50]}")
            _fetch(page_url, page_path)
            time.sleep(REQUEST_DELAY_S)
        page_html = page_path.read_text(encoding="utf-8")
        page = parse_comic_page(page_html, comic_id)

        journal_abstract = ""
        journal_title = ""
        if fetch_journals and page["has_journal"]:
            journal_path = JOURNALS_DIR / f"{comic_id}.html"
            journal_url = f"https://phdcomics.com/archive_journal.php?n={comic_id}"
            if force or not journal_path.is_file():
                try:
                    _fetch(journal_url, journal_path)
                    time.sleep(REQUEST_DELAY_S)
                except RuntimeError as exc:
                    print(f"  journal fetch failed: {exc}")
            if journal_path.is_file():
                journal = parse_journal_page(journal_path.read_text(encoding="utf-8"))
                journal_abstract = journal.get("abstract", "")
                journal_title = journal.get("journal_title", "")

        record = {
            **item,
            "page_title": page["title"],
            "image_url": page["image_url"],
            "image_alt": page["image_alt"],
            "has_journal": page["has_journal"],
            "journal_title": journal_title,
            "journal_abstract": journal_abstract,
            "comic_url": page_url,
            "source": "phdcomics_most_popular",
        }
        records_by_id[comic_id] = record
        completed.add(comic_id)
        manifest["completed_ids"] = sorted(completed, key=int)
        save_manifest(manifest)

        ordered = [records_by_id[t["comic_id"]] for t in targets if t["comic_id"] in records_by_id]
        COMICS_JSONL.parent.mkdir(parents=True, exist_ok=True)
        COMICS_JSONL.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + "\n",
            encoding="utf-8",
        )

    return [records_by_id[t["comic_id"]] for t in targets if t["comic_id"] in records_by_id]


def write_geek_jokes_jsonl(records: list[dict], dest: Path = GEEK_JOKES_PATH) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [to_training_row(r) for r in records]
    dest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} training rows -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Scrape only first N comics")
    parser.add_argument("--force", action="store_true", help="Re-fetch cached HTML")
    parser.add_argument("--no-journals", action="store_true", help="Skip journal abstract pages")
    parser.add_argument(
        "--listing-only",
        action="store_true",
        help="Parse listing and write geek_jokes.jsonl from cached pages only",
    )
    args = parser.parse_args()

    listing = scrape_listing(force=args.force)
    print(f"parsed {len(listing)} comics from most popular")

    if args.listing_only:
        records = []
        for item in listing[: args.limit or len(listing)]:
            comic_id = item["comic_id"]
            page_path = PAGES_DIR / f"{comic_id}.html"
            if not page_path.is_file():
                raise SystemExit(f"missing cached page for comic {comic_id}; run without --listing-only")
            page = parse_comic_page(page_path.read_text(encoding="utf-8"), comic_id)
            journal_abstract = ""
            journal_title = ""
            journal_path = JOURNALS_DIR / f"{comic_id}.html"
            if journal_path.is_file():
                journal = parse_journal_page(journal_path.read_text(encoding="utf-8"))
                journal_abstract = journal.get("abstract", "")
                journal_title = journal.get("journal_title", "")
            records.append(
                {
                    **item,
                    "page_title": page["title"],
                    "image_url": page["image_url"],
                    "image_alt": page["image_alt"],
                    "has_journal": page["has_journal"],
                    "journal_title": journal_title,
                    "journal_abstract": journal_abstract,
                    "comic_url": f"https://phdcomics.com/comics/archive.php?comicid={comic_id}",
                    "source": "phdcomics_most_popular",
                }
            )
    else:
        records = scrape_comics(
            listing,
            limit=args.limit,
            force=args.force,
            fetch_journals=not args.no_journals,
        )

    write_geek_jokes_jsonl(records)
    print(f"raw records -> {COMICS_JSONL}")
    print(f"manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
