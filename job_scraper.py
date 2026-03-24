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
    r"(?i)^initiativ",
    r"(?i)^spontanbewerbung",
    r"(?i)^ihre\s+initiativbewerbung",
    r"(?i)^blind\s+application",
    r"(?i)^offene\s+bewerbung",
    r"(?i)^unsolicited\s+application",
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
            print(f"  [Fetch] JS-Rendering fehlgeschlagen (HTTP {e.response.status_code}) → Retry ohne JS", file=sys.stderr)
            return fetch_html(url, render_js=False)
        raise


# Data attributes commonly used by JS frameworks instead of href
_DATA_LINK_ATTRS = ["data-href", "data-url", "data-link", "data-target",
                    "data-detail-url", "data-job-url", "data-path"]

# Regex to extract URLs from onclick handlers
_ONCLICK_URL_RE = re.compile(
    r"""(?:location\.href|window\.(?:open|location)|navigate|href)\s*[=(]\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)


def _find_job_lists_in(obj: object, depth: int = 0) -> list[list]:
    """Recursively find lists that look like job listings in arbitrary nested JSON."""
    if depth > 8:
        return []
    if isinstance(obj, list):
        if _looks_like_job_list(obj):
            return [obj]
        nested: list[list] = []
        for item in obj:
            nested.extend(_find_job_lists_in(item, depth + 1))
        return nested
    if isinstance(obj, dict):
        found: list[list] = []
        for v in obj.values():
            found.extend(_find_job_lists_in(v, depth + 1))
        return found
    return []


def _extract_nextjs_jobs(raw_html: str, karriereseite: str) -> Optional[list[JobInfo]]:
    """Extract job listings from Next.js __NEXT_DATA__ embedded JSON.

    Next.js sites embed all page data in a <script id="__NEXT_DATA__"> block.
    clean_html strips all scripts, so this must run on the raw HTML.
    Returns None if no __NEXT_DATA__ found or no job-like data inside.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return None

    try:
        data = json.loads(script.string)
    except (json.JSONDecodeError, TypeError):
        return None

    job_lists = _find_job_lists_in(data)
    if not job_lists:
        return None

    # Take the largest candidate list (most likely the full job listing)
    best = max(job_lists, key=len)
    print(f"  [Next.js] __NEXT_DATA__: {len(best)} Einträge gefunden", file=sys.stderr)

    results = []
    for item in best:
        if not isinstance(item, dict):
            continue
        title = (
            item.get("title") or item.get("name") or item.get("stellentitel")
            or item.get("jobtitle") or item.get("position")
        )
        if not title or _is_generic_application(str(title)):
            continue

        # Try common URL / slug patterns
        detail_url = (
            item.get("url") or item.get("link")
            or item.get("apply-url") or item.get("applyUrl")
            or item.get("ad-url") or item.get("adUrl")
        )
        if not detail_url:
            slug = item.get("slug") or item.get("uri") or item.get("path")
            if slug:
                detail_url = urljoin(karriereseite, slug)

        # Try to extract aufgaben/profil from any description field
        description = item.get("description") or item.get("content") or item.get("text") or ""
        aufgaben, profil = _parse_description_html(str(description)) if description else ([], [])

        # Solique/SABAG-style: textblocks["text block"] list with internal-name keys
        if not aufgaben and not profil:
            textblocks = item.get("textblocks") or {}
            blocks = textblocks.get("text block") or []
            if isinstance(blocks, dict):
                blocks = [blocks]
            block_map: dict[str, str] = {}
            for block in blocks:
                if isinstance(block, dict):
                    key = (block.get("internal-name") or "").strip()
                    text = str(block.get("text") or "").strip()
                    if key and text:
                        block_map[key] = text
            if block_map:
                raw_aufgaben = block_map.get("Aufgaben") or block_map.get("aufgaben") or ""
                raw_profil = block_map.get("Profil") or block_map.get("profil") or ""
                aufgaben = [li.strip("- ").strip() for li in raw_aufgaben.split("- ") if li.strip("- ").strip()]
                profil = [li.strip("- ").strip() for li in raw_profil.split("- ") if li.strip("- ").strip()]

        results.append(JobInfo(
            karriereseite=karriereseite,
            stellen_url=detail_url or None,
            stellentitel=str(title),
            aufgaben=aufgaben,
            profil=profil,
        ))

    if results:
        with_links = sum(1 for j in results if j.stellen_url)
        print(
            f"  [Next.js] {len(results)} Stelle(n) extrahiert "
            f"({with_links} mit Detail-Link)",
            file=sys.stderr,
        )
    return results if results else None


def _extract_jsonld_jobs(raw_html: str, karriereseite: str) -> Optional[list[JobInfo]]:
    """Extract jobs from JSON-LD structured data (schema.org/JobPosting).

    Many job sites embed complete job data in <script type="application/ld+json">.
    This is the highest-quality source: no Claude needed, no link guessing.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    postings: list[dict] = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Normalise: unwrap @graph arrays
        if isinstance(data, dict) and "@graph" in data:
            data = data["@graph"]
        if not isinstance(data, list):
            data = [data]

        for item in data:
            if not isinstance(item, dict):
                continue
            typ = item.get("@type", "")
            # Accept both string and list type declarations
            types = [typ] if isinstance(typ, str) else typ
            if "JobPosting" in types:
                postings.append(item)

    if not postings:
        return None

    print(f"  [JSON-LD] {len(postings)} JobPosting(s) gefunden", file=sys.stderr)
    results = []
    for p in postings:
        title = p.get("title") or p.get("name")
        if not title or _is_generic_application(title):
            continue

        # Extract aufgaben + profil from description HTML
        description = p.get("description") or ""
        aufgaben, profil = _parse_description_html(description)

        detail_url = p.get("url") or p.get("identifier", {}).get("value") if isinstance(p.get("identifier"), dict) else None

        results.append(JobInfo(
            karriereseite=karriereseite,
            stellen_url=detail_url or None,
            stellentitel=title,
            aufgaben=aufgaben,
            profil=profil,
        ))

    return results if results else None


def _parse_description_html(html: str) -> tuple[list[str], list[str]]:
    """Heuristically split a job description into aufgaben and profil bullet points."""
    if not html:
        return [], []

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    aufgaben: list[str] = []
    profil: list[str] = []
    current: list[str] = []  # buffer before first section detected

    _AUFGABEN_HEADERS = re.compile(
        r"(?i)(aufgaben|tätigkeiten|deine aufgabe|was dich erwartet|responsibilities|your tasks|aufgabenbereich)"
    )
    _PROFIL_HEADERS = re.compile(
        r"(?i)(profil|anforderungen|qualifikation|was du mitbringst|requirements|your profile|das bringst du mit)"
    )

    mode = None
    for line in lines:
        if _AUFGABEN_HEADERS.search(line):
            mode = "aufgaben"
            continue
        if _PROFIL_HEADERS.search(line):
            mode = "profil"
            continue
        if mode == "aufgaben":
            aufgaben.append(line)
        elif mode == "profil":
            profil.append(line)

    # Limit to reasonable length
    return aufgaben[:20], profil[:20]


def clean_html(raw_html: str, max_chars: int = 80_000) -> str:
    """Strip noise tags and extract readable text, preserving all link signals inline."""
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # 1) Inline <a href> — standard links
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(strip=True)
        if href and text:
            a.replace_with(f"{text} [{href}]")
        elif href:
            a.replace_with(f"[{href}]")

    # 2) Inline data-href / data-url / ... on non-anchor elements
    #    e.g. <div data-href="/jobs/123">Job Title</div>
    for attr in _DATA_LINK_ATTRS:
        for tag in soup.find_all(attrs={attr: True}):
            url_val = tag.get(attr, "").strip()
            if not url_val or url_val.startswith("javascript:"):
                continue
            text = tag.get_text(strip=True)
            if text:
                tag.replace_with(f"{text} [{url_val}]")

    # 3) Extract URLs from onclick handlers
    #    e.g. <div onclick="location.href='/jobs/123'">Job Title</div>
    for tag in soup.find_all(onclick=True):
        onclick = tag.get("onclick", "")
        match = _ONCLICK_URL_RE.search(onclick)
        if match:
            url_val = match.group(1)
            text = tag.get_text(strip=True)
            if text:
                tag.replace_with(f"{text} [{url_val}]")

    # Try to find main content area, fall back to body if too little text
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=lambda x: x and "job" in x.lower())
        or soup.find(class_=lambda x: x and any(
            kw in " ".join(x).lower()
            for kw in ("job", "stelle", "position", "career", "karriere", "vacancy")
        ))
    )

    if main:
        if len(main.get_text(strip=True)) < 200:
            main = soup.find("body") or soup
    else:
        main = soup.find("body") or soup

    text = main.get_text(separator="\n", strip=True)
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


# ---------------------------------------------------------------------------
# Playwright: network-interception fallback for JS-heavy sites
# ---------------------------------------------------------------------------
# Triggered automatically when the regular pipeline finds titles but no links
# on an overview page. Playwright intercepts every XHR/fetch response and
# looks for job data in the captured JSON payloads.
#
# Playwright is an optional dependency — if not installed the scraper still
# works, just without this fallback.

def _fetch_with_playwright(url: str) -> tuple[str, list[dict]]:
    """Load URL in a real headless browser, capture all JSON API responses.

    Returns (rendered_html, [{"url": ..., "data": ...}, ...]).
    Raises RuntimeError if Playwright is not installed.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        raise RuntimeError(
            "Playwright nicht installiert. "
            "Ausführen: pip install playwright && playwright install chromium"
        )

    captured: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def handle_response(response):
            if response.request.resource_type in ("image", "stylesheet", "font", "media"):
                return
            ct = response.headers.get("content-type", "")
            if "json" not in ct:
                return
            try:
                body = response.json()
                if body:
                    captured.append({"url": response.url, "data": body})
            except Exception:
                pass

        page.on("response", handle_response)

        try:
            page.goto(url, wait_until="networkidle", timeout=30_000)
        except PWTimeout:
            # networkidle timed out — DOM is likely ready, XHR may still run
            pass

        html = page.content()
        browser.close()

    return html, captured


def _looks_like_job_list(items: list) -> bool:
    """Heuristic: does this list look like a list of job postings?"""
    if not isinstance(items, list) or len(items) < 1:
        return False
    sample = items[0] if isinstance(items[0], dict) else {}
    job_fields = {"title", "name", "jobtitle", "stellentitel", "position"}
    return bool(job_fields & {k.lower() for k in sample.keys()})


def _extract_jobs_from_api_responses(
    api_responses: list[dict], karriereseite: str
) -> Optional[list[JobInfo]]:
    """Search captured Playwright API responses for job listing data.

    Returns JobInfo list if anything useful is found, else None.
    """
    candidates: list[dict] = []

    for resp in api_responses:
        data = resp.get("data")
        source_url = resp.get("url", "")

        # Unwrap common envelope shapes
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            raw_list = None
            for key in ("jobs", "jobpostings", "items", "results", "data", "entries"):
                if isinstance(data.get(key), list):
                    raw_list = data[key]
                    break
            if raw_list is None:
                continue
        else:
            continue

        if not _looks_like_job_list(raw_list):
            continue

        print(
            f"  [Playwright] Job-API erkannt: {source_url} "
            f"({len(raw_list)} Einträge)",
            file=sys.stderr,
        )
        candidates.extend(raw_list)

    if not candidates:
        return None

    results: list[JobInfo] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        # Normalise title
        title = (
            item.get("title") or item.get("name") or item.get("jobtitle")
            or item.get("stellentitel") or item.get("position")
        )
        if not title or _is_generic_application(str(title)):
            continue

        # Normalise detail URL
        detail_url = (
            item.get("url") or item.get("detailUrl") or item.get("link")
            or item.get("applyUrl") or item.get("jobUrl")
        )

        # Normalise aufgaben / profil from description if present
        description = item.get("description") or item.get("content") or ""
        aufgaben, profil = _parse_description_html(str(description))

        results.append(JobInfo(
            karriereseite=karriereseite,
            stellen_url=detail_url or None,
            stellentitel=str(title),
            aufgaben=aufgaben,
            profil=profil,
        ))

    print(f"  [Playwright] {len(results)} Jobs aus API extrahiert", file=sys.stderr)
    return results if results else None


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
    _playwright_attempted: bool = False,
) -> list[JobInfo]:
    """Full pipeline: fetch → clean → extract for a single URL.

    karriereseite: set when this is a detail-page call from an overview page.
    Returns a list of JobInfo objects.
    """
    is_detail_call = karriereseite is not None
    effective_karriereseite = karriereseite or url

    label = "[Detail]" if is_detail_call else "[Übersicht]"
    print(f"\n{'─'*60}", file=sys.stderr)
    print(f"{label} {url}", file=sys.stderr)

    try:
        print(f"  [Fetch] ScraperAPI {'mit' if render_js else 'ohne'} JS-Rendering...", file=sys.stderr)
        raw_html = fetch_html(url, render_js=render_js)
        print(f"  [Fetch] HTML erhalten: {len(raw_html):,} Zeichen", file=sys.stderr)

        # Next.js __NEXT_DATA__ — try to get full job list before scripts are stripped
        # Only on overview pages (not detail calls), to avoid losing data the scraper
        # would otherwise miss due to JS-rendered accordions / lazy-loading.
        if not is_detail_call:
            nextjs_jobs = _extract_nextjs_jobs(raw_html, effective_karriereseite)
            if nextjs_jobs is not None:
                jobs_with_links = [j for j in nextjs_jobs if j.stellen_url]
                if jobs_with_links:
                    print(f"  [Next.js] Folge {len(jobs_with_links)} Detail-Link(s) mit Claude...", file=sys.stderr)
                    all_detail_jobs: list[JobInfo] = []
                    for i, j in enumerate(jobs_with_links, 1):
                        print(f"  [Next.js] Link {i}/{len(jobs_with_links)}: {j.stellen_url}", file=sys.stderr)
                        detail_jobs = scrape_jobs(
                            j.stellen_url, client,
                            karriereseite=effective_karriereseite,
                            render_js=render_js,
                        )
                        all_detail_jobs.extend(detail_jobs)
                    all_detail_jobs = [j for j in all_detail_jobs
                                       if not _is_generic_application(j.stellentitel or "")]
                    if all_detail_jobs:
                        print(f"  [Next.js] ✓ {len(all_detail_jobs)} Stelle(n) via Detail-Seiten", file=sys.stderr)
                        return all_detail_jobs
                    # Detail-Seiten lieferten nichts → fall through to Claude
                    print("  [Next.js] Detail-Seiten leer → falle zurück auf Claude", file=sys.stderr)
                else:
                    # No detail URLs (accordion-only page) — return __NEXT_DATA__ results directly
                    print(f"  [Next.js] Keine Detail-Links → gebe {len(nextjs_jobs)} Titel zurück", file=sys.stderr)
                    return nextjs_jobs

        # Generic pipeline: clean HTML → Claude
        content = clean_html(raw_html)
        print(f"  [Clean] Bereinigter Text: {len(content):,} Zeichen", file=sys.stderr)

        if not content.strip():
            print("  [!] Kein verwertbarer Inhalt nach Bereinigung", file=sys.stderr)
            return [JobInfo(karriereseite=effective_karriereseite, stellen_url=url if is_detail_call else None,
                            fehler="Kein verwertbarer Inhalt nach Bereinigung")]

        print("  [Claude] Extrahiere Stelleninfos...", file=sys.stderr)
        extracted = extract_job_info(content, client)

        seitentyp = extracted.get("seitentyp", "einzelstelle")
        stellen_roh = extracted.get("stellen") or []
        print(f"  [Claude] Seitentyp: {seitentyp} | {len(stellen_roh)} Stelle(n) erkannt", file=sys.stderr)

        # Post-processing: filter generic application entries
        stellen = [s for s in stellen_roh if not _is_generic_application(s.get("stellentitel", "") or "")]
        gefiltert = len(stellen_roh) - len(stellen)
        if gefiltert:
            print(f"  [Filter] {gefiltert} Initiativ-/Spontanbewerbung(en) entfernt → {len(stellen)} verbleiben", file=sys.stderr)

        # --- Overview page: follow individual job links ---
        if seitentyp == "uebersicht" and not is_detail_call:
            job_links = extracted.get("job_links") or []
            print(
                f"  [Übersicht] {len(stellen)} Stelle(n) im Listing | {len(job_links)} Einzel-Link(s) erkannt",
                file=sys.stderr,
            )

            if job_links:
                print(f"  [Übersicht] Folge {len(job_links)} Detail-Link(s)...", file=sys.stderr)
                all_detail_jobs: list[JobInfo] = []
                for i, link in enumerate(job_links, 1):
                    abs_link = urljoin(url, link)
                    print(f"  [Übersicht] Link {i}/{len(job_links)}: {abs_link}", file=sys.stderr)
                    detail_jobs = scrape_jobs(
                        abs_link, client, karriereseite=url, render_js=render_js
                    )
                    all_detail_jobs.extend(detail_jobs)

                # Filter generic applications from detail results too
                vorher = len(all_detail_jobs)
                all_detail_jobs = [j for j in all_detail_jobs if not _is_generic_application(j.stellentitel or "")]
                if len(all_detail_jobs) < vorher:
                    print(f"  [Filter] {vorher - len(all_detail_jobs)} generische Einträge aus Detailseiten entfernt", file=sys.stderr)
                print(f"  [Übersicht] ✓ {len(all_detail_jobs)} Stelle(n) nach Detail-Scraping", file=sys.stderr)
                return all_detail_jobs

            # No links found — try Playwright before giving up
            if not _playwright_attempted:
                print(
                    "  [Playwright] Keine Links via ScraperAPI → starte Playwright-Fallback...",
                    file=sys.stderr,
                )
                try:
                    pw_html, pw_api = _fetch_with_playwright(url)
                    print(f"  [Playwright] HTML: {len(pw_html):,} Zeichen | {len(pw_api)} API-Response(s) abgefangen", file=sys.stderr)

                    # Re-run JSON-LD on the Playwright-rendered DOM
                    pw_jsonld = _extract_jsonld_jobs(pw_html, effective_karriereseite)
                    if pw_jsonld:
                        print(f"  [Playwright] JSON-LD: {len(pw_jsonld)} Stelle(n) gefunden", file=sys.stderr)
                        return pw_jsonld

                    # Search captured XHR/fetch responses for job data
                    pw_api_jobs = _extract_jobs_from_api_responses(pw_api, effective_karriereseite)
                    if pw_api_jobs:
                        print(f"  [Playwright] XHR-API: {len(pw_api_jobs)} Stelle(n) gefunden", file=sys.stderr)
                        return pw_api_jobs

                    # Re-run Claude pipeline on Playwright-rendered HTML
                    print("  [Playwright] Kein XHR-Treffer → Claude auf Playwright-HTML...", file=sys.stderr)
                    pw_content = clean_html(pw_html)
                    print(f"  [Playwright] Bereinigter Text: {len(pw_content):,} Zeichen", file=sys.stderr)
                    if pw_content.strip():
                        pw_extracted = extract_job_info(pw_content, client)
                        pw_links = pw_extracted.get("job_links") or []
                        pw_stellen = [
                            s for s in (pw_extracted.get("stellen") or [])
                            if not _is_generic_application(s.get("stellentitel", "") or "")
                        ]
                        print(f"  [Playwright/Claude] {len(pw_stellen)} Stelle(n) | {len(pw_links)} Link(s)", file=sys.stderr)
                        if pw_links:
                            all_detail_jobs: list[JobInfo] = []
                            for i, link in enumerate(pw_links, 1):
                                abs_link = urljoin(url, link)
                                print(f"  [Playwright] Link {i}/{len(pw_links)}: {abs_link}", file=sys.stderr)
                                all_detail_jobs.extend(
                                    scrape_jobs(abs_link, client, karriereseite=url,
                                                render_js=render_js, _playwright_attempted=True)
                                )
                            return [j for j in all_detail_jobs
                                    if not _is_generic_application(j.stellentitel or "")]
                        if pw_stellen:
                            return [_build_job(effective_karriereseite, None, s) for s in pw_stellen]
                except RuntimeError as e:
                    print(f"  [Playwright] Nicht verfügbar: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"  [Playwright] Fehler: {type(e).__name__}: {e}", file=sys.stderr)

            # All methods exhausted — last resort: b-ite DOM/API for title guarantee
            bite_fallback = _parse_bite_jobs(raw_html, effective_karriereseite)
            if bite_fallback:
                print(f"  [b-ite] Fallback: {len(bite_fallback)} Titel extrahiert (kein Stellendetail)", file=sys.stderr)
                return bite_fallback

            if not stellen:
                print("  [!] Keine Stellen gefunden (alle Methoden erschöpft)", file=sys.stderr)
                return [JobInfo(karriereseite=effective_karriereseite, fehler="Keine Stellen gefunden")]
            print(f"  [!] Keine Einzel-Links → gebe {len(stellen)} Listing-Titel zurück (ohne Details)", file=sys.stderr)
            return [_build_job(effective_karriereseite, None, s) for s in stellen]

        # --- Single job page (or detail call) ---
        if not stellen:
            print("  [!] Einzelseite: keine Stelle extrahiert", file=sys.stderr)
            return [JobInfo(karriereseite=effective_karriereseite,
                            stellen_url=url if is_detail_call else None,
                            fehler="Keine Stellen gefunden")]

        jobs = [_build_job(effective_karriereseite, url if is_detail_call else None, s) for s in stellen]
        titel = jobs[0].stellentitel or "(Titel unbekannt)"
        aufg = len(jobs[0].aufgaben)
        prof = len(jobs[0].profil)
        print(f"  [Detail] ✓ '{titel}' | {aufg} Aufgabe(n) | {prof} Profil-Punkt(e)", file=sys.stderr)
        return jobs

    except requests.exceptions.RequestException as e:
        print(f"  [!] Fetch-Fehler ({type(e).__name__}): {_sanitize_error(str(e))}", file=sys.stderr)
        return [JobInfo(karriereseite=effective_karriereseite,
                        stellen_url=url if is_detail_call else None,
                        fehler=f"HTTP-Fehler: {_sanitize_error(str(e))}")]
    except json.JSONDecodeError as e:
        print(f"  [!] JSON-Parse-Fehler: {e}", file=sys.stderr)
        return [JobInfo(karriereseite=effective_karriereseite,
                        stellen_url=url if is_detail_call else None,
                        fehler=f"JSON-Parsing fehlgeschlagen: {e}")]
    except anthropic.APIError as e:
        print(f"  [!] Claude API-Fehler ({type(e).__name__}): {e}", file=sys.stderr)
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
