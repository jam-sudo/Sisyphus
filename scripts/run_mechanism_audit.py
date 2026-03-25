#!/usr/bin/env python3
"""Phase 2 Holdout Mechanism Audit.

For each holdout drug, annotate:
- P-gp (ABCB1) substrate/inhibitor
- BCRP (ABCG2) substrate
- OATP1B1/1B3 substrate
- OAT/OCT substrate
- UGT substrate (which isoforms)
- Fe > 0.3 (renal dominant)

Cross-reference with engine fold errors to identify which missing
mechanisms explain the largest errors.

Output: TSV table + summary statistics.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Data paths ──────────────────────────────────────────────────────────
_DATA_DIR = _ROOT / "data"
_DRUGBANK_DIR = _DATA_DIR / "drugbank"
_REF_DIR = _DATA_DIR / "reference"

# ── Transporter name normalization ─────────────────────────────────────
_TRANSPORTER_NORM: dict[str, str] = {
    "ATP-dependent translocase ABCB1": "P_gp",
    "Broad substrate specificity ATP-binding cassette transporter ABCG2": "BCRP",
    "Solute carrier organic anion transporter family member 1B1": "OATP1B1",
    "Solute carrier organic anion transporter family member 1B3": "OATP1B3",
    "Solute carrier family 22 member 1": "OCT1",
    "Solute carrier family 22 member 2": "OCT2",
    "Solute carrier family 22 member 6": "OAT1",
    "Solute carrier family 22 member 8": "OAT3",
    "Solute carrier family 22 member 7": "OAT2",
    "Solute carrier family 22 member 11": "OAT4",
    "Multidrug and toxin extrusion protein 1": "MATE1",
    "Multidrug and toxin extrusion protein 2": "MATE2",
}

# ── UGT name normalization ────────────────────────────────────────────
_UGT_NORM: dict[str, str] = {
    "UDP-glucuronosyltransferase 2B7": "UGT2B7",
    "UDP-glucuronosyltransferase 1A1": "UGT1A1",
    "UDP-glucuronosyltransferase 1A4": "UGT1A4",
    "UDP-glucuronosyltransferase 1A9": "UGT1A9",
    "UDP-glucuronosyltransferase 1A3": "UGT1A3",
    "UDP-glucuronosyltransferase 1A6": "UGT1A6",
    "UDP-glucuronosyltransferase 1A8": "UGT1A8",
    "UDP-glucuronosyltransferase 1A10": "UGT1A10",
    "UDP-glucuronosyltransferase 2B4": "UGT2B4",
    "UDP-glucuronosyltransferase 2B10": "UGT2B10",
    "UDP-glucuronosyltransferase 2B15": "UGT2B15",
}


@dataclass
class DrugAudit:
    """Mechanism audit record for a single holdout drug."""
    name: str
    smiles: str
    dose_mg: float
    cmax_obs: float
    compound_type: str = ""
    dbid: str = ""

    # Engine results
    engine_cmax: float = 0.0
    engine_fold: float = 0.0  # pred/obs
    meta_cmax: float = 0.0
    meta_fold: float = 0.0

    # Transporter annotations (substrate only)
    p_gp: bool = False
    bcrp: bool = False
    oatp1b1: bool = False
    oatp1b3: bool = False
    oat: bool = False  # OAT1/2/3
    oct: bool = False  # OCT1/2
    mate: bool = False

    # UGT annotations (substrate only)
    ugt_isoforms: list[str] = field(default_factory=list)

    # CYP annotations
    cyp_isoforms: list[str] = field(default_factory=list)

    # Renal
    fe_text: str = ""  # raw text from DrugBank route of elimination

    # AD flags
    ad_flags: tuple[str, ...] = ()

    @property
    def is_ugt_substrate(self) -> bool:
        return len(self.ugt_isoforms) > 0

    @property
    def is_transporter_substrate(self) -> bool:
        return self.p_gp or self.bcrp or self.oatp1b1 or self.oatp1b3

    @property
    def expected_mechanism(self) -> str:
        """Hypothesize which missing mechanism explains engine error."""
        mechanisms = []
        fold = self.engine_fold
        if fold == 0:
            return "engine_failed"

        # Over-prediction (fold > 1) + P-gp substrate → missing efflux
        if fold > 1.5 and self.p_gp:
            mechanisms.append("P-gp_efflux")
        if fold > 1.5 and self.bcrp:
            mechanisms.append("BCRP_efflux")

        # UGT substrate → fm misallocation
        if self.is_ugt_substrate:
            mechanisms.append("UGT_fm")

        # OATP substrate → hepatic uptake
        if self.oatp1b1 or self.oatp1b3:
            mechanisms.append("OATP_uptake")

        # OAT/OCT → renal secretion
        if self.oat or self.oct:
            mechanisms.append("renal_secretion")

        return ",".join(mechanisms) if mechanisms else "none"


def load_drugbank_name_to_id() -> dict[str, str]:
    """Load drug name → DrugBank ID mapping."""
    mapping = {}
    path = _DRUGBANK_DIR / "drugs.csv"
    if not path.exists():
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip().lower()
            dbid = row.get("drugbank_id", "").strip()
            if name and dbid:
                mapping[name] = dbid
    return mapping


def load_transporter_annotations() -> dict[str, dict[str, set[str]]]:
    """Load DrugBank ID → {transporter_tag: {actions}}."""
    result: dict[str, dict[str, set[str]]] = {}
    path = _DRUGBANK_DIR / "transporter_annotations.csv"
    if not path.exists():
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dbid = row.get("drugbank_id", "")
            tname = row.get("transporter_name", "")
            actions = row.get("actions", "")
            tag = _TRANSPORTER_NORM.get(tname)
            if tag and dbid:
                result.setdefault(dbid, {}).setdefault(tag, set())
                for a in actions.split(","):
                    a = a.strip()
                    if a:
                        result[dbid][tag].add(a)
    return result


def load_ugt_annotations() -> dict[str, list[str]]:
    """Load DrugBank ID → [UGT isoforms] (substrate only)."""
    result: dict[str, list[str]] = {}
    path = _DRUGBANK_DIR / "enzyme_annotations.csv"
    if not path.exists():
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "substrate" not in row.get("actions", ""):
                continue
            dbid = row.get("drugbank_id", "")
            ename = row.get("enzyme_name", "")
            ugt_tag = _UGT_NORM.get(ename)
            if ugt_tag and dbid:
                result.setdefault(dbid, [])
                if ugt_tag not in result[dbid]:
                    result[dbid].append(ugt_tag)
    return result


def load_cyp_annotations() -> dict[str, list[str]]:
    """Load DrugBank ID → [CYP isoforms] (substrate only)."""
    _CYP_NORM = {
        "Cytochrome P450 3A4": "CYP3A4",
        "Cytochrome P450 2D6": "CYP2D6",
        "Cytochrome P450 1A2": "CYP1A2",
        "Cytochrome P450 2C9": "CYP2C9",
        "Cytochrome P450 2E1": "CYP2E1",
        "Cytochrome P450 3A5": "CYP3A5",
        "Cytochrome P450 2C19": "CYP2C19",
        "Cytochrome P450 2C8": "CYP2C8",
        "Cytochrome P450 2B6": "CYP2B6",
    }
    result: dict[str, list[str]] = {}
    path = _DRUGBANK_DIR / "enzyme_annotations.csv"
    if not path.exists():
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "substrate" not in row.get("actions", ""):
                continue
            dbid = row.get("drugbank_id", "")
            ename = row.get("enzyme_name", "")
            cyp_tag = _CYP_NORM.get(ename)
            if cyp_tag and dbid:
                result.setdefault(dbid, [])
                if cyp_tag not in result[dbid]:
                    result[dbid].append(cyp_tag)
    return result


def run_predictions(holdout_refs: list) -> dict[str, tuple]:
    """Run pipeline predictions, return {drug_name: (engine_cmax, meta_cmax, compound_type, ad_flags)}."""
    from sisyphus.pipeline.predict import predict

    results = {}
    for i, ref in enumerate(holdout_refs):
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            engine_cmax = result.engine_pk.cmax.mean if result.engine_pk else 0.0
            meta_cmax = result.pk.cmax.mean

            # Get compound_type from chemistry layer
            from sisyphus.predict.chemistry import compute_profile
            profile = compute_profile(ref.smiles)
            compound_type = profile.compound_type

            results[ref.name] = (engine_cmax, meta_cmax, compound_type, result.ad_flags)
            print(
                f"  [{i+1}/{len(holdout_refs)}] {ref.name}: "
                f"eng={engine_cmax:.4f} meta={meta_cmax:.4f} obs={ref.cmax_obs:.4f} "
                f"type={compound_type}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(holdout_refs)}] {ref.name}: FAILED ({e})", file=sys.stderr)
            results[ref.name] = (0.0, 0.0, "", ())
    return results


def main():
    from sisyphus.validation.reference import load_reference
    from sisyphus.predict.drugbank import DrugBankLookup

    print("Loading reference data...", file=sys.stderr)
    refs = load_reference()
    holdout_refs = [r for r in refs if r.in_holdout]
    print(f"  {len(holdout_refs)} holdout drugs", file=sys.stderr)

    # DrugBank lookup (SMILES → DrugBank ID)
    print("Loading DrugBank data...", file=sys.stderr)
    db = DrugBankLookup()
    db._ensure_loaded()

    transporter_data = load_transporter_annotations()
    ugt_data = load_ugt_annotations()
    cyp_data = load_cyp_annotations()

    print(f"  Transporters: {len(transporter_data)} drugs annotated", file=sys.stderr)
    print(f"  UGT: {len(ugt_data)} drugs annotated", file=sys.stderr)
    print(f"  CYP: {len(cyp_data)} drugs annotated", file=sys.stderr)

    # Run engine predictions
    print("Running predictions...", file=sys.stderr)
    predictions = run_predictions(holdout_refs)

    # Build audit records
    audits: list[DrugAudit] = []
    for ref in holdout_refs:
        a = DrugAudit(
            name=ref.name,
            smiles=ref.smiles,
            dose_mg=ref.dose_mg,
            cmax_obs=ref.cmax_obs,
        )

        # DrugBank ID lookup
        dbid = db.lookup(ref.smiles)
        a.dbid = dbid or ""

        if dbid:
            # Transporter annotations
            trans = transporter_data.get(dbid, {})
            a.p_gp = "substrate" in trans.get("P_gp", set())
            a.bcrp = "substrate" in trans.get("BCRP", set())
            a.oatp1b1 = "substrate" in trans.get("OATP1B1", set())
            a.oatp1b3 = "substrate" in trans.get("OATP1B3", set())
            a.oat = any(
                "substrate" in trans.get(t, set())
                for t in ("OAT1", "OAT2", "OAT3", "OAT4")
            )
            a.oct = any(
                "substrate" in trans.get(t, set())
                for t in ("OCT1", "OCT2")
            )
            a.mate = any(
                "substrate" in trans.get(t, set())
                for t in ("MATE1", "MATE2")
            )

            # UGT annotations
            a.ugt_isoforms = ugt_data.get(dbid, [])

            # CYP annotations
            a.cyp_isoforms = cyp_data.get(dbid, [])

        # Engine predictions
        eng_cmax, meta_cmax, ctype, ad_flags = predictions.get(ref.name, (0.0, 0.0, "", ()))
        a.engine_cmax = eng_cmax
        a.meta_cmax = meta_cmax
        a.compound_type = ctype
        a.ad_flags = ad_flags
        if eng_cmax > 0 and ref.cmax_obs > 0:
            a.engine_fold = eng_cmax / ref.cmax_obs
        if meta_cmax > 0 and ref.cmax_obs > 0:
            a.meta_fold = meta_cmax / ref.cmax_obs

        audits.append(a)

    # ── Output TSV ──────────────────────────────────────────────────────
    header = [
        "Drug", "Type", "Cmax_obs", "Eng_Cmax", "Eng_fold", "Meta_fold",
        "P-gp", "BCRP", "OATP1B1", "OATP1B3", "OAT", "OCT",
        "UGT", "UGT_isoforms", "CYP_isoforms",
        "AD_flags", "Expected_mechanism",
    ]
    print("\t".join(header))

    for a in sorted(audits, key=lambda x: abs(np.log10(x.engine_fold)) if x.engine_fold > 0 else 99, reverse=True):
        print("\t".join([
            a.name,
            a.compound_type,
            f"{a.cmax_obs:.4f}",
            f"{a.engine_cmax:.4f}",
            f"{a.engine_fold:.2f}" if a.engine_fold > 0 else "FAIL",
            f"{a.meta_fold:.2f}" if a.meta_fold > 0 else "FAIL",
            "Y" if a.p_gp else "",
            "Y" if a.bcrp else "",
            "Y" if a.oatp1b1 else "",
            "Y" if a.oatp1b3 else "",
            "Y" if a.oat else "",
            "Y" if a.oct else "",
            "Y" if a.is_ugt_substrate else "",
            ";".join(a.ugt_isoforms),
            ";".join(a.cyp_isoforms),
            ",".join(a.ad_flags) if a.ad_flags else "",
            a.expected_mechanism,
        ]))

    # ── Summary statistics ─────────────────────────────────────────────
    print("\n\n# ── SUMMARY ──", file=sys.stderr)

    n_total = len(audits)
    n_pgp = sum(1 for a in audits if a.p_gp)
    n_bcrp = sum(1 for a in audits if a.bcrp)
    n_oatp = sum(1 for a in audits if a.oatp1b1 or a.oatp1b3)
    n_ugt = sum(1 for a in audits if a.is_ugt_substrate)
    n_oat_oct = sum(1 for a in audits if a.oat or a.oct)

    print(f"Total holdout drugs: {n_total}", file=sys.stderr)
    print(f"P-gp substrates:    {n_pgp} ({100*n_pgp/n_total:.0f}%)", file=sys.stderr)
    print(f"BCRP substrates:    {n_bcrp} ({100*n_bcrp/n_total:.0f}%)", file=sys.stderr)
    print(f"OATP substrates:    {n_oatp} ({100*n_oatp/n_total:.0f}%)", file=sys.stderr)
    print(f"UGT substrates:     {n_ugt} ({100*n_ugt/n_total:.0f}%)", file=sys.stderr)
    print(f"OAT/OCT substrates: {n_oat_oct} ({100*n_oat_oct/n_total:.0f}%)", file=sys.stderr)

    # Engine fold errors by mechanism
    for mech_name, mech_filter in [
        ("P-gp substrates", lambda a: a.p_gp),
        ("UGT substrates", lambda a: a.is_ugt_substrate),
        ("OATP substrates", lambda a: a.oatp1b1 or a.oatp1b3),
        ("No transporter/UGT", lambda a: not a.p_gp and not a.is_ugt_substrate and not (a.oatp1b1 or a.oatp1b3)),
    ]:
        subset = [a for a in audits if mech_filter(a) and a.engine_fold > 0]
        if not subset:
            continue
        folds = np.array([a.engine_fold for a in subset])
        afes = np.power(10, np.mean(np.abs(np.log10(folds))))
        n_over = sum(1 for f in folds if f > 1.5)
        n_under = sum(1 for f in folds if f < 1/1.5)
        print(
            f"\n{mech_name} (N={len(subset)}): AAFE={afes:.3f}, "
            f"over={n_over}, under={n_under}",
            file=sys.stderr,
        )
        for a in sorted(subset, key=lambda x: abs(np.log10(x.engine_fold)), reverse=True)[:5]:
            print(
                f"  {a.name}: eng_fold={a.engine_fold:.2f} "
                f"UGT={';'.join(a.ugt_isoforms) or '-'} "
                f"CYP={';'.join(a.cyp_isoforms) or '-'}",
                file=sys.stderr,
            )

    # Drugs where UGT could help most (UGT substrate + large engine error)
    print("\n# ── UGT IMPACT CANDIDATES ──", file=sys.stderr)
    ugt_drugs = [a for a in audits if a.is_ugt_substrate and a.engine_fold > 0]
    for a in sorted(ugt_drugs, key=lambda x: abs(np.log10(x.engine_fold)), reverse=True):
        direction = "OVER" if a.engine_fold > 1 else "UNDER"
        print(
            f"  {a.name}: eng_fold={a.engine_fold:.2f} ({direction}) "
            f"UGT={';'.join(a.ugt_isoforms)} CYP={';'.join(a.cyp_isoforms) or 'none'} "
            f"type={a.compound_type}",
            file=sys.stderr,
        )

    # Drugs where P-gp could help most (P-gp substrate + engine over-predicts)
    print("\n# ── P-gp IMPACT CANDIDATES ──", file=sys.stderr)
    pgp_drugs = [a for a in audits if a.p_gp and a.engine_fold > 1.5]
    for a in sorted(pgp_drugs, key=lambda x: x.engine_fold, reverse=True):
        print(
            f"  {a.name}: eng_fold={a.engine_fold:.2f} (OVER) "
            f"type={a.compound_type}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
