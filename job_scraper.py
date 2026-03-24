"""
Job scraper: fetches career pages via ScraperAPI, extracts structured job info via Claude.
Dynamic by design — Claude adapts to any site structure without hardcoded parsers.

Usage:
    python job_scraper.py <url> [<url2> ...]
    python job_scraper.py --file urls.txt
    python job_scraper.py --help
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
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
    "profil": ["Anforderung 1", "Anforderung 2"],
    "ansprechpartner": {
      "name": "Name oder null",
      "titel": "z.B. HR Manager oder null",
      "email": "email@beispiel.de oder null",
      "telefon": "Telefonnummer oder null"
    },
    "unternehmen": "Unternehmensname oder null",
    "standort": "Standort oder null",
    "beschaeftigungsart": "Vollzeit/Teilzeit/Remote oder null",
    "hinweise": "Sonstige wichtige Infos oder null"
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
      "profil": [],
      "ansprechpartner": {"name": null, "titel": null, "email": null, "telefon": null},
      "unternehmen": "Unternehmensname oder null",
      "standort": "Standort oder null",
      "beschaeftigungsart": "Art der Stelle oder null",
      "hinweise": "Weitere verfügbare Infos oder null"
    }
  ]
}

Regeln:
- Extrahiere NUR was tatsächlich auf der Seite steht — erfinde nichts
- Bei Übersichtsseiten: job_links sind die URLs der Einzelstellenseiten (aus den [URL]-Angaben im Text)
  Nur Links aufnehmen, die wirklich zu einer Stellendetailseite führen (nicht Filterseiten, nicht die aktuelle Seite selbst)
- Bei Übersichtsseiten: ALLE gefundenen Stellen auflisten, auch wenn Details fehlen
- IGNORIERE und ÜBERSPRINGE vollständig Einträge wie "Initiativbewerbung", "Spontanbewerbung",
  "Blind Application", "Offene Bewerbung" oder ähnliche allgemeine Bewerbungsoptionen ohne
  konkreten Stellentitel — diese sind keine echten Stellenanzeigen und sollen nicht extrahiert werden
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
    ansprechpartner: Optional[dict] = None
    unternehmen: Optional[str] = None
    standort: Optional[str] = None
    beschaeftigungsart: Optional[str] = None
    hinweise: Optional[str] = None
    fehler: Optional[str] = None


def fetch_html(url: str, render_js: bool = True) -> str:
    """Fetch page HTML via ScraperAPI. Falls back to non-JS rendering on error."""
    params = {
        "api_key": SCRAPER_API_KEY,
        "url": url,
        "render": "true" if render_js else "false",
    }
    if render_js:
        # Wait 5 s after initial load so JS-heavy job boards have time to render
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

    # Extract main content areas preferentially
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=lambda x: x and "job" in x.lower())
        or soup.find(class_=lambda x: x and any(
            kw in " ".join(x).lower() for kw in ("job", "stelle", "position", "career")
        ))
        or soup.find("body")
        or soup
    )

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


def _build_job(karriereseite: str, stellen_url: Optional[str], stelle: dict) -> JobInfo:
    effective_stellen_url = stellen_url if stellen_url and stellen_url != karriereseite else None
    return JobInfo(
        karriereseite=karriereseite,
        stellen_url=effective_stellen_url,
        stellentitel=stelle.get("stellentitel"),
        aufgaben=stelle.get("aufgaben") or [],
        profil=stelle.get("profil") or [],
        ansprechpartner=stelle.get("ansprechpartner"),
        unternehmen=stelle.get("unternehmen"),
        standort=stelle.get("standort"),
        beschaeftigungsart=stelle.get("beschaeftigungsart"),
        hinweise=stelle.get("hinweise"),
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

        print("  Bereinige HTML...", file=sys.stderr)
        content = clean_html(raw_html)

        if not content.strip():
            return [JobInfo(karriereseite=effective_karriereseite, stellen_url=url if is_detail_call else None,
                            fehler="Kein verwertbarer Inhalt nach Bereinigung")]

        print("  Extrahiere Stelleninfos via Claude...", file=sys.stderr)
        extracted = extract_job_info(content, client)

        seitentyp = extracted.get("seitentyp", "einzelstelle")
        stellen = extracted.get("stellen") or []

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
        print(f"  ✗ Fetch-Fehler: {e}", file=sys.stderr)
        return [JobInfo(karriereseite=effective_karriereseite,
                        stellen_url=url if is_detail_call else None,
                        fehler=f"HTTP-Fehler: {e}")]
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
    print(f"Unternehmen:   {job.unternehmen or '–'}")
    print(f"Standort:      {job.standort or '–'}")
    print(f"Art:           {job.beschaeftigungsart or '–'}")

    if job.aufgaben:
        print("\nAufgaben:")
        for a in job.aufgaben:
            print(f"  • {a}")

    if job.profil:
        print("\nProfil:")
        for p in job.profil:
            print(f"  • {p}")

    if job.ansprechpartner:
        ap = job.ansprechpartner
        has_contact = any(v for v in ap.values() if v)
        if has_contact:
            print("\nAnsprechpartner:")
            if ap.get("name"):
                line = f"  {ap['name']}"
                if ap.get("titel"):
                    line += f" ({ap['titel']})"
                print(line)
            if ap.get("email"):
                print(f"  E-Mail: {ap['email']}")
            if ap.get("telefon"):
                print(f"  Tel.: {ap['telefon']}")

    if job.hinweise:
        print(f"\nHinweise: {job.hinweise}")


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
  python job_scraper.py https://jobs.firma.de/karriere
  python job_scraper.py --file urls.txt --output ergebnisse.json
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

    # Collect URLs
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

    results = []
    all_jobs: list[JobInfo] = []
    for url in urls:
        jobs = scrape_jobs(url, client, render_js=render_js)
        all_jobs.extend(jobs)
        results.extend(asdict(j) for j in jobs)

    # Output
    output_json = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        total = len(all_jobs)
        print(f"\n✓ {total} Stelle(n) gespeichert: {args.output}", file=sys.stderr)
        for job in all_jobs:
            print_job(job)
    else:
        for job in all_jobs:
            print_job(job)
        print("\n" + "=" * 60 + "\nJSON-Ausgabe:\n" + "=" * 60)
        print(output_json)


if __name__ == "__main__":
    main()
