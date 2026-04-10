"""Systematic FDA label Cmax extractor via openFDA API.

For each drug name, fetches the most recent FDA prescribing label and
extracts Cmax values from the clinical_pharmacology section using
regex patterns specific to FDA label formatting conventions.

Usage:
    python3 scripts/fda_cmax_extractor.py <drug1> <drug2> ...
    python3 scripts/fda_cmax_extractor.py --file drugs.txt

Output: JSON with per-drug Cmax extracts (value, unit, single-dose vs SS).

Built to expand prospective validation beyond N=10. Supplements but
does not replace manual curation — FDA labels vary in format and
often require dose/AR disambiguation per drug.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_API = "https://api.fda.gov/drug/label.json"
_UNITS = {'ng/mL': 0.001, 'ng/ml': 0.001, 'mcg/mL': 1.0, 'mcg/ml': 1.0,
          'μg/mL': 1.0, 'mg/L': 1.0, 'g/L': 1000.0}

_PATTERNS = [
    # "C max,ss (ng/mL) a 778" — FDA table format
    (r'C\s*max(?P<ss>,ss|,SS)?\s*\(?(?P<unit>ng/mL|mcg/mL|μg/mL|mg/L)\)?\s*[a-z]?\s*(?P<val>[\d,]+\.?\d*)', True),
    # "Cmax of 133 ng/mL" — prose format
    (r'C\s*max(?P<ss>,ss|,SS)?[^.]{0,60}?(?P<val>[\d,]+\.?\d*)\s*(?P<unit>ng/mL|mcg/mL|μg/mL|mg/L)', False),
]


def fetch_label(name: str, timeout: int = 15) -> dict | None:
    """Fetch the most recent FDA label for a drug by generic/brand name."""
    for field in ['generic_name', 'brand_name']:
        q = f'openfda.{field}:"{name}"'
        url = f"{_API}?search={urllib.parse.quote(q)}&limit=1"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            if 'results' in data:
                return data['results'][0]
        except Exception:
            continue
    return None


def extract_cmax(label: dict) -> list[dict]:
    """Extract Cmax values from label's clinical_pharmacology section.

    Returns list of {mg_L, unit, raw_value, is_steady_state, context}.
    """
    if not label:
        return []
    cp_text = ' '.join(label.get('clinical_pharmacology', ['']))
    cp_text += ' ' + ' '.join(label.get('clinical_pharmacology_table', ['']))
    cp_text = re.sub(r'<[^>]+>', ' ', cp_text)  # strip HTML

    results = []
    for pattern, _unit_first in _PATTERNS:
        for m in re.finditer(pattern, cp_text):
            try:
                val = float(m.group('val').replace(',', ''))
                unit = m.group('unit')
                if unit not in _UNITS or val < 0.01 or val > 100000:
                    continue
                mg_L = val * _UNITS[unit]
                is_ss = bool(m.group('ss'))
                results.append({
                    'mg_L': round(mg_L, 4),
                    'raw_value': val,
                    'unit': unit,
                    'is_steady_state': is_ss,
                    'context': m.group(0)[:80],
                })
            except Exception:
                continue
    return results


def extract_dose_hints(label: dict) -> list[str]:
    """Extract dose-related hints from dosage_and_administration section."""
    if not label:
        return []
    dosing = ' '.join(label.get('dosage_and_administration', ['']))
    dosing = re.sub(r'<[^>]+>', ' ', dosing)
    # Look for "X mg" patterns
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*mg', dosing[:1000])
    return list(dict.fromkeys(matches))[:5]  # first 5 unique


def process_drugs(names: list[str], delay: float = 0.3) -> dict:
    """Process a list of drug names. Returns {name: extraction_result}."""
    output = {}
    for name in names:
        label = fetch_label(name)
        if label is None:
            output[name] = {'status': 'no_label_found'}
        else:
            cmax = extract_cmax(label)
            doses = extract_dose_hints(label)
            output[name] = {
                'status': 'ok' if cmax else 'no_cmax_match',
                'cmax_extracts': cmax,
                'dose_hints_mg': doses,
            }
        time.sleep(delay)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('drugs', nargs='*', help='drug names (generic)')
    parser.add_argument('--file', help='file with one drug per line')
    parser.add_argument('--output', default='fda_cmax_extracts.json')
    parser.add_argument('--delay', type=float, default=0.3)
    args = parser.parse_args()

    drugs = list(args.drugs)
    if args.file:
        drugs.extend(Path(args.file).read_text().strip().split('\n'))
    drugs = [d.strip() for d in drugs if d.strip()]
    if not drugs:
        print("Usage: fda_cmax_extractor.py <drug1> <drug2> ... or --file <path>")
        sys.exit(1)

    results = process_drugs(drugs, delay=args.delay)
    Path(args.output).write_text(json.dumps(results, indent=2))

    n_ok = sum(1 for r in results.values() if r.get('status') == 'ok')
    print(f"Extracted Cmax for {n_ok}/{len(drugs)} drugs")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
