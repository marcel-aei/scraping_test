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

EXTRACTION_PROMPT = """Du bist ein Spezialist für die Analyse von Stellenanzeigen.
Analysiere den folgenden HTML-Inhalt einer Karriereseite und extrahiere alle Stelleninformationen.

Gib deine Antwort als JSON-Objekt zurück mit folgender Struktur:
{
  "stellentitel": "Exakter Titel der Stelle",
  "aufgaben": ["Aufgabe 1", "Aufgabe 2", ...],
  "profil": ["Anforderung 1", "Anforderung 2", ...],
  "ansprechpartner": {
    "name": "Vor- und Nachname oder null",
    "titel": "z.B. HR Manager oder null",
    "email": "email@beispiel.de oder null",
    "telefon": "Telefonnummer oder null"
  },
  "unternehmen": "Unternehmensname oder null",
  "standort": "Standort/Stadt oder null",
  "beschaeftigungsart": "z.B. Vollzeit, Teilzeit, Remote oder null",
  "hinweise": "Wichtige Informationen, die nicht in die anderen Felder passen, oder null"
}

Regeln:
- Extrahiere NUR was tatsächlich auf der Seite steht — erfinde nichts
- Bei mehreren Stellen auf einer Seite: extrahiere die prominenteste/erste vollständige Stelle
- Wenn ein Feld nicht gefunden wird, setze den Wert auf null (bei Listen: leeres Array [])
- Aufgaben und Profil als einzelne, klare Stichpunkte (keine Oberkategorien)
- Antwort NUR als reines JSON, kein Markdown, keine Erklärung

HTML-Inhalt:
"""


@dataclass
class JobInfo:
    url: str
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
    try:
        response = requests.get(SCRAPER_API_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        if render_js and e.response is not None and e.response.status_code in (429, 500):
            print(f"  [!] JS rendering fehlgeschlagen, versuche ohne JS...", file=sys.stderr)
            return fetch_html(url, render_js=False)
        raise


def clean_html(raw_html: str, max_chars: int = 40_000) -> str:
    """Strip noise tags and extract readable text to reduce token usage."""
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup(NOISE_TAGS := _NOISE_TAGS):
        tag.decompose()

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


def extract_job_info(content: str, url: str, client: anthropic.Anthropic) -> dict:
    """Use Claude to extract structured job info from cleaned page content."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
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


def scrape_job(url: str, client: anthropic.Anthropic) -> JobInfo:
    """Full pipeline: fetch → clean → extract for a single URL."""
    job = JobInfo(url=url)
    print(f"\n→ Verarbeite: {url}", file=sys.stderr)

    try:
        print("  Fetching HTML...", file=sys.stderr)
        raw_html = fetch_html(url)

        print("  Bereinige HTML...", file=sys.stderr)
        content = clean_html(raw_html)

        if not content.strip():
            job.fehler = "Kein verwertbarer Inhalt nach Bereinigung"
            return job

        print("  Extrahiere Stelleninfos via Claude...", file=sys.stderr)
        extracted = extract_job_info(content, url, client)

        job.stellentitel = extracted.get("stellentitel")
        job.aufgaben = extracted.get("aufgaben") or []
        job.profil = extracted.get("profil") or []
        job.ansprechpartner = extracted.get("ansprechpartner")
        job.unternehmen = extracted.get("unternehmen")
        job.standort = extracted.get("standort")
        job.beschaeftigungsart = extracted.get("beschaeftigungsart")
        job.hinweise = extracted.get("hinweise")

        print(f"  ✓ Stelle gefunden: {job.stellentitel or '(Titel unbekannt)'}", file=sys.stderr)

    except requests.exceptions.RequestException as e:
        job.fehler = f"HTTP-Fehler: {e}"
        print(f"  ✗ Fetch-Fehler: {e}", file=sys.stderr)
    except json.JSONDecodeError as e:
        job.fehler = f"JSON-Parsing fehlgeschlagen: {e}"
        print(f"  ✗ JSON-Fehler: {e}", file=sys.stderr)
    except anthropic.APIError as e:
        job.fehler = f"Claude API-Fehler: {e}"
        print(f"  ✗ Claude-Fehler: {e}", file=sys.stderr)

    return job


def print_job(job: JobInfo) -> None:
    """Pretty-print a single job result to stdout."""
    print("\n" + "=" * 60)
    print(f"URL: {job.url}")

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
  python job_scraper.py https://jobs.firma.de/stelle-1 https://jobs.firma.de/stelle-2
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

    results = []
    for url in urls:
        job = scrape_job(url, client)
        results.append(asdict(job))

    # Output
    output_json = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\n✓ Ergebnisse gespeichert: {args.output}", file=sys.stderr)
    else:
        print("\n" + "=" * 60 + "\nJSON-Ausgabe:\n" + "=" * 60)
        print(output_json)

    # Pretty-print to stderr for readability when writing to file
    if args.output:
        for job_dict in results:
            print_job(JobInfo(**job_dict))


if __name__ == "__main__":
    main()
