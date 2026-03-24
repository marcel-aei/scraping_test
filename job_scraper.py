"""
Job scraper: fetches career pages via ScraperAPI, extracts structured job info via Claude.
Dynamic by design — Claude adapts to any site structure without hardcoded parsers.

Only extracts: Stellentitel, Aufgaben, Profil, Link (no personal data).

Usage:
    python job_scraper.py <url> [<url2> ...]
    python job_scraper.py --file urls.txt
    python job_scraper.py --diff ergebnisse_vorher.json ergebnisse_nachher.json
    python job_scraper.py --help
"""

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

import anthropic
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")

SCRAPER_API_URL = "http://api.scraperapi.com"

# Tags that add noise but no content
_NOISE_TAGS = [
    "script", "style", "noscript", "iframe", "svg", "img",
    "header", "footer", "nav", "aside",
]

# Patterns for generic application entries (not real job postings)
_GENERIC_APPLICATION_PATTERNS = [
    r"(?i)^initiativbewerbung$",
    r"(?i)^ihre\s+initiativbewerbung$",
    r"(?i)^spontanbewerbung$",
    r"(?i)^blind\s+application$",
    r"(?i)^offene\s+bewerbung$",
    r"(?i)^unsolicited\s+application$",
    r"(?i)^initiativ",
]

EXTRACTION_PROMPT = """Du bist ein Spezialist für die Analyse von Karriereseiten und Stellenanzeigen.

Analysiere den folgenden Seiteninhalt und bestimme zuerst den Seitentyp:

A) EINZELNE STELLENANZEIGE: Die Seite zeigt eine konkrete Stelle mit Details (Aufgaben, Profil, etc.)
B) ÜBERSICHTSSEITE: Die Seite listet mehrere Stellen auf (Karriereübersicht, Jobboard, etc.)

Gib deine Antwort als JSON zurück:

Bei Typ A (Einzelstelle):
{
  "seitentyp": "einzelstelle",
  "stellen": [{
    "stellentitel": "Exakter Titel",
    "aufgaben": ["Aufgabe 1", "Aufgabe 2"],
    "profil": ["Anforderung 1", "Anforderung 2"]
  }]
}

Bei Typ B (Übersichtsseite):
{
  "seitentyp": "uebersicht",
  "job_links": [
    "URL oder relativer Pfad zur Einzel-Stellenseite (aus den [Link]-Angaben im Text entnehmen)"
  ],
  "stellen": [
    {
      "stellentitel": "Jobtitel",
      "aufgaben": [],
      "profil": []
    }
  ]
}

Regeln:
- Extrahiere NUR was tatsächlich auf der Seite steht — erfinde nichts
- Bei Übersichtsseiten: job_links sind die URLs der Einzelstellenseiten (aus den [URL]-Angaben im Text)
  Nur Links aufnehmen, die wirklich zu einer Stellendetailseite führen (nicht Filterseiten, nicht die aktuelle Seite selbst)
- Bei Übersichtsseiten: ALLE gefundenen Stellen auflisten, auch wenn Details fehlen
- Wenn ein Feld nicht vorhanden ist: null (bei Listen: [])
- Aufgaben und Profil: einzelne, klare Stichpunkte
- Antwort NUR als reines JSON, kein Markdown, keine Erklärung

Seiteninhalt:
"""


@dataclass
class JobInfo:
    karriereseite: str = ""
    stellen_url: Optional[str] = None
    stellentitel: Optional[str] = None
    aufgaben: list = field(default_factory=list)
    profil: list = field(default_factory=list)
    fehler: Optional[str] = None


def _sanitize_error(msg: str) -> str:
    """Remove API keys from error messages."""
    if SCRAPER_API_KEY:
        msg = msg.replace(SCRAPER_API_KEY, "***")
    return msg


def _is_generic_application(title: str) -> bool:
    """Check if a job title is a generic application entry (not a real posting)."""
    if not title:
        return False
    return any(re.match(pat, title.strip()) for pat in _GENERIC_APPLICATION_PATTERNS)


def fetch_html(url: str, render_js: bool = True) -> str:
    """Fetch page HTML via ScraperAPI. Falls back to non-JS rendering on error."""
    params = {
        "api_key": SCRAPER_API_KEY,
        "url": url,
        "render": "true" if render_js else "false",
    }
    if render_js:
        params["wait"] = "5000"
    try:
        response = requests.get(SCRAPER_API_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        if render_js and e.response is not None and e.response.status_code in (429, 500):
            print(f"  [!] JS rendering fehlgeschlagen, versuche ohne JS...", file=sys.stderr)
            return fetch_html(url, render_js=False)
        raise


def clean_html(raw_html: str, max_chars: int = 80_000) -> str:
    """Strip noise tags and extract readable text, preserving link hrefs inline."""
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # Inline link hrefs so Claude can see and extract URLs
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(strip=True)
        if href and text:
            a.replace_with(f"{text} [{href}]")
        elif href:
            a.replace_with(f"[{href}]")

    # Try to find main content area, but fall back to body if too little text
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=lambda x: x and "job" in x.lower())
        or soup.find(class_=lambda x: x and any(
            kw in " ".join(x).lower() for kw in ("job", "stelle", "position", "career", "karriere", "vacancy")
        ))
    )

    # If main content area has too little text, fall back to body
    if main:
        main_text = main.get_text(strip=True)
        if len(main_text) < 200:
            main = soup.find("body") or soup
    else:
        main = soup.find("body") or soup

    text = main.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) > max_chars:
        print(
            f"  [i] Inhalt gekürzt: {len(cleaned):,} → {max_chars:,} Zeichen",
            file=sys.stderr,
        )
        cleaned = cleaned[:max_chars]

    return cleaned


def extract_job_info(content: str, client: anthropic.Anthropic) -> dict:
    """Use Claude to extract structured job info from cleaned page content."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT + content,
            }
        ],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code blocks if present
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines if not line.startswith("```")
        )

    return json.loads(raw_text)


# ---------------------------------------------------------------------------
# b-ite Career Suite: special direct parser
# ---------------------------------------------------------------------------
# b-ite renders job entries as <div class="bite-jobs--entry"> with NO href.
# Links are constructed on-click via JS and are not accessible from the HTML.
# We detect b-ite, extract titles directly from DOM (no Claude needed),
# and try the b-ite public API for detail URLs + job content.
#
# b-ite API (v5): https://api.b-ite.com/v5/jobpostings/{company-slug}
# Detail page:    https://jobs.{company-domain}/jobposting/{hash}

_BITE_API_BASE = "https://api.b-ite.com/v5/jobpostings"


def _detect_bite_company(soup: BeautifulSoup) -> Optional[str]:
    """Return the b-ite company slug if this page uses b-ite Career Suite."""
    tag = soup.find(attrs={"data-bite-jobs-api-listing": True})
    if not tag:
        return None
    # Format: "schlueter-baumaschinen:main-listing"
    value = tag["data-bite-jobs-api-listing"]
    return value.split(":")[0]


def _fetch_bite_api(company_slug: str) -> Optional[list[dict]]:
    """Try to fetch job listings from the b-ite public API.
    Returns list of raw job dicts or None on failure.
    """
    url = f"{_BITE_API_BASE}/{company_slug}"
    try:
        resp = requests.get(url, timeout=15, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        # API returns either a list or {"jobpostings": [...]}
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("jobpostings", "jobs", "data", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
        return None
    except Exception as e:
        print(f"  [b-ite] API nicht erreichbar ({url}): {e}", file=sys.stderr)
        return None


def _parse_bite_jobs(raw_html: str, karriereseite: str) -> Optional[list[JobInfo]]:
    """Parse b-ite Career Suite pages.

    Strategy:
    1. Try the b-ite public API for job data incl. detail URLs
    2. Fall back to DOM-parsing for titles only (no Claude needed, no links)
    """
    soup = BeautifulSoup(raw_html, "lxml")
    company_slug = _detect_bite_company(soup)
    if not company_slug:
        return None

    print(f"  [b-ite] Erkannt: {company_slug}", file=sys.stderr)

    # --- Try b-ite API first ---
    api_jobs = _fetch_bite_api(company_slug)
    if api_jobs:
        print(f"  [b-ite] API: {len(api_jobs)} Jobs gefunden", file=sys.stderr)
        results = []
        for job in api_jobs:
            title = job.get("title") or job.get("name") or job.get("stellentitel")
            job_id = job.get("id") or job.get("hash") or job.get("jobpostingId")
            # Construct detail URL from known pattern
            detail_url = None
            if job_id:
                # Try to derive the jobs subdomain from karriereseite domain
                # e.g. www.wir-sind-schlueter.de → jobs.schlueter-baumaschinen.de is unknown,
                # but the API response might include a url field
                detail_url = (
                    job.get("url") or job.get("detailUrl") or job.get("link")
                    or f"https://api.b-ite.com/v5/jobpostings/{company_slug}/{job_id}"
                )
            if title and not _is_generic_application(title):
                results.append(JobInfo(
                    karriereseite=karriereseite,
                    stellen_url=detail_url,
                    stellentitel=title,
                    aufgaben=job.get("tasks") or job.get("aufgaben") or [],
                    profil=job.get("requirements") or job.get("profil") or [],
                ))
        if results:
            return results

    # --- Fallback: extract titles from DOM ---
    print(
        "  [b-ite] Fallback: extrahiere Titel aus DOM (keine Links verfügbar)",
        file=sys.stderr,
    )
    results = []
    for entry in soup.find_all(class_="bite-jobs--entry--title"):
        title = entry.get_text(strip=True)
        if title and not _is_generic_application(title):
            results.append(JobInfo(
                karriereseite=karriereseite,
                stellen_url=None,
                stellentitel=title,
                aufgaben=[],
                profil=[],
            ))

    if results:
        print(f"  [b-ite] {len(results)} Stellen-Titel extrahiert (kein Stellendetail)", file=sys.stderr)
        return results

    return None


def _build_job(karriereseite: str, stellen_url: Optional[str], stelle: dict) -> JobInfo:
    effective_stellen_url = stellen_url if stellen_url and stellen_url != karriereseite else None
    return JobInfo(
        karriereseite=karriereseite,
        stellen_url=effective_stellen_url,
        stellentitel=stelle.get("stellentitel"),
        aufgaben=stelle.get("aufgaben") or [],
        profil=stelle.get("profil") or [],
    )


def scrape_jobs(
    url: str,
    client: anthropic.Anthropic,
    karriereseite: Optional[str] = None,
    render_js: bool = True,
) -> list[JobInfo]:
    """Full pipeline: fetch → clean → extract for a single URL.

    karriereseite: set when this is a detail-page call from an overview page.
    Returns a list of JobInfo objects.
    """
    is_detail_call = karriereseite is not None
    effective_karriereseite = karriereseite or url

    print(f"\n→ Verarbeite: {url}", file=sys.stderr)

    try:
        print("  Fetching HTML...", file=sys.stderr)
        raw_html = fetch_html(url, render_js=render_js)

        # b-ite Career Suite: handle before generic pipeline
        if not is_detail_call:
            bite_jobs = _parse_bite_jobs(raw_html, effective_karriereseite)
            if bite_jobs is not None:
                return bite_jobs

        print("  Bereinige HTML...", file=sys.stderr)
        content = clean_html(raw_html)

        if not content.strip():
            return [JobInfo(karriereseite=effective_karriereseite, stellen_url=url if is_detail_call else None,
                            fehler="Kein verwertbarer Inhalt nach Bereinigung")]

        print("  Extrahiere Stelleninfos via Claude...", file=sys.stderr)
        extracted = extract_job_info(content, client)

        seitentyp = extracted.get("seitentyp", "einzelstelle")
        stellen = extracted.get("stellen") or []

        # Post-processing: filter generic application entries
        stellen = [s for s in stellen if not _is_generic_application(s.get("stellentitel", "") or "")]
        if not stellen and extracted.get("stellen"):
            print("  [i] Alle Stellen waren Initiativ-/Spontanbewerbungen → übersprungen", file=sys.stderr)

        # --- Overview page: follow individual job links ---
        if seitentyp == "uebersicht" and not is_detail_call:
            job_links = extracted.get("job_links") or []
            print(
                f"  ✓ Übersichtsseite: {len(stellen)} Stelle(n) gefunden, "
                f"{len(job_links)} Einzel-Link(s) erkannt",
                file=sys.stderr,
            )

            if job_links:
                print(f"  → Folge {len(job_links)} Einzel-Links für Details...", file=sys.stderr)
                all_detail_jobs: list[JobInfo] = []
                for link in job_links:
                    abs_link = urljoin(url, link)
                    detail_jobs = scrape_jobs(
                        abs_link, client, karriereseite=url, render_js=render_js
                    )
                    all_detail_jobs.extend(detail_jobs)

                # Filter generic applications from detail results too
                all_detail_jobs = [j for j in all_detail_jobs if not _is_generic_application(j.stellentitel or "")]
                return all_detail_jobs

            # No links found — return overview-level data (no detail available)
            if not stellen:
                return [JobInfo(karriereseite=effective_karriereseite, fehler="Keine Stellen gefunden")]
            return [_build_job(effective_karriereseite, None, s) for s in stellen]

        # --- Single job page (or detail call) ---
        if not stellen:
            return [JobInfo(karriereseite=effective_karriereseite,
                            stellen_url=url if is_detail_call else None,
                            fehler="Keine Stellen gefunden")]

        jobs = [_build_job(effective_karriereseite, url if is_detail_call else None, s) for s in stellen]
        print(
            f"  ✓ Stelle gefunden: {jobs[0].stellentitel or '(Titel unbekannt)'}",
            file=sys.stderr,
        )
        return jobs

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Fetch-Fehler: {_sanitize_error(str(e))}", file=sys.stderr)
        return [JobInfo(karriereseite=effective_karriereseite,
                        stellen_url=url if is_detail_call else None,
                        fehler=f"HTTP-Fehler: {_sanitize_error(str(e))}")]
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON-Fehler: {e}", file=sys.stderr)
        return [JobInfo(karriereseite=effective_karriereseite,
                        stellen_url=url if is_detail_call else None,
                        fehler=f"JSON-Parsing fehlgeschlagen: {e}")]
    except anthropic.APIError as e:
        print(f"  ✗ Claude-Fehler: {e}", file=sys.stderr)
        return [JobInfo(karriereseite=effective_karriereseite,
                        stellen_url=url if is_detail_call else None,
                        fehler=f"Claude API-Fehler: {e}")]


def print_job(job: JobInfo) -> None:
    """Pretty-print a single job result to stdout."""
    print("\n" + "=" * 60)
    print(f"Karriereseite: {job.karriereseite}")
    if job.stellen_url:
        print(f"Stellen-URL:   {job.stellen_url}")

    if job.fehler:
        print(f"FEHLER: {job.fehler}")
        return

    print(f"Stelle:        {job.stellentitel or '–'}")

    if job.aufgaben:
        print("\nAufgaben:")
        for a in job.aufgaben:
            print(f"  • {a}")

    if job.profil:
        print("\nProfil:")
        for p in job.profil:
            print(f"  • {p}")




# ---------------------------------------------------------------------------
# Summary per career page
# ---------------------------------------------------------------------------

def print_summary(all_jobs: list[JobInfo]) -> None:
    """Print a summary grouped by career page."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for job in all_jobs:
        grouped[job.karriereseite].append(job)

    print(f"\n{'=' * 70}", file=sys.stderr)
    print("ZUSAMMENFASSUNG", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    for site, jobs in grouped.items():
        real_jobs = [j for j in jobs if not j.fehler]
        error_jobs = [j for j in jobs if j.fehler]
        total = len(real_jobs)
        aufgaben_leer = sum(1 for j in real_jobs if not j.aufgaben)
        profil_leer = sum(1 for j in real_jobs if not j.profil)
        beide_leer = sum(1 for j in real_jobs if not j.aufgaben and not j.profil)

        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"Karriereseite: {site}", file=sys.stderr)
        print(f"  Stellen gesamt:          {total}", file=sys.stderr)
        if total > 0:
            print(f"  Aufgaben leer:           {aufgaben_leer} / {total}  ({aufgaben_leer/total*100:.1f}%)", file=sys.stderr)
            print(f"  Profil leer:             {profil_leer} / {total}  ({profil_leer/total*100:.1f}%)", file=sys.stderr)
            print(f"  Aufgaben UND Profil leer:{beide_leer} / {total}  ({beide_leer/total*100:.1f}%)", file=sys.stderr)
        print(f"\n  Gefundene Stellen:", file=sys.stderr)
        for j in real_jobs:
            tags = []
            if not j.aufgaben:
                tags.append("Aufgaben leer")
            if not j.profil:
                tags.append("Profil leer")
            tag_str = f" [{'] ['.join(tags)}]" if tags else ""
            print(f"    - {j.stellentitel or '(kein Titel)'}{tag_str}", file=sys.stderr)
        for j in error_jobs:
            print(f"    ✗ FEHLER: {j.fehler}", file=sys.stderr)


def validate_env() -> None:
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not SCRAPER_API_KEY:
        missing.append("SCRAPER_API_KEY")
    if missing:
        print(
            f"Fehler: Folgende Umgebungsvariablen fehlen: {', '.join(missing)}\n"
            "Tipp: Kopiere .env.example nach .env und trage deine Keys ein.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Stelleninformationen von Karriereseiten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python job_scraper.py https://example.com/jobs/software-engineer
  python job_scraper.py --file urls.txt --output ergebnisse.json
  python job_scraper.py --diff alt.json neu.json
        """,
    )
    parser.add_argument("urls", nargs="*", help="Karriereseiten-URLs")
    parser.add_argument(
        "--file", "-f",
        metavar="DATEI",
        help="Textdatei mit URLs (eine pro Zeile)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DATEI",
        help="JSON-Ausgabedatei (Standard: stdout)",
    )
    parser.add_argument(
        "--no-js",
        action="store_true",
        help="JS-Rendering deaktivieren (schneller, für statische Seiten)",
    )

    args = parser.parse_args()

    # --- Scrape mode ---
    urls = list(args.urls)
    if args.file:
        try:
            with open(args.file) as f:
                file_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            urls.extend(file_urls)
        except FileNotFoundError:
            print(f"Fehler: Datei nicht gefunden: {args.file}", file=sys.stderr)
            sys.exit(1)

    if not urls:
        parser.print_help()
        sys.exit(1)

    validate_env()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    render_js = not args.no_js

    all_jobs: list[JobInfo] = []
    for url in urls:
        jobs = scrape_jobs(url, client, render_js=render_js)
        all_jobs.extend(jobs)

    # Build output with metadata
    results = [asdict(j) for j in all_jobs]
    output_data = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "urls_count": len(urls),
            "stellen_count": sum(1 for j in all_jobs if not j.fehler),
            "fehler_count": sum(1 for j in all_jobs if j.fehler),
        },
        "stellen": results,
    }

    output_json = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\n✓ {len(all_jobs)} Stelle(n) gespeichert: {args.output}", file=sys.stderr)

    # Always print summary and detailed output to stderr/stdout
    print_summary(all_jobs)
    for job in all_jobs:
        print_job(job)

    if not args.output:
        print("\n" + "=" * 60 + "\nJSON-Ausgabe:\n" + "=" * 60)
        print(output_json)


if __name__ == "__main__":
    main()
