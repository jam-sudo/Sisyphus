"""Rodgers–Rowland tissue partitioning (J Pharm Sci 94:1259; 95:1238)."""

from __future__ import annotations

from dataclasses import dataclass

from sisyphus.core import Distribution


@dataclass(frozen=True)
class RRTissue:
    """Volume fractions, acidic phospholipid mg/g, and tissue:plasma protein ratios."""

    few: Distribution
    fiw: Distribution
    fnl: Distribution
    fnp: Distribution
    ap_mg_g: Distribution
    albumin_ratio: Distribution
    lipoprotein_ratio: Distribution


def _tissue(values: tuple[float, ...]) -> RRTissue:
    return RRTissue(*(Distribution(value, 0.0) for value in values))


# Rodgers et al. 2005 Table 1 (PMID 15858854);
# Rodgers & Rowland 2006 Table 1 (PMID 16639716).
# Values transcribed in https://pmc.ncbi.nlm.nih.gov/articles/PMC11196289/ Table III
# and https://dash.harvard.edu/bitstreams/643906fa-6801-45e0-8574-8b9015da8048/download
# Table 2. Each row: fEW, fIW, fNL, fNP, AP mg/g, albumin ratio, lipoprotein ratio.
_RR_ROWS: dict[str, RRTissue] = {
    "adipose_tissue": _tissue((.135, .017, .853, .0016, .40, .049, .068)),
    "bone_tissue": _tissue((.100, .346, .0174, .0016, .67, .100, .050)),
    "brain": _tissue((.162, .620, .0391, .0015, .40, .048, .041)),
    "gut_wall": _tissue((.282, .475, .0375, .0124, 2.41, .158, .141)),
    "heart": _tissue((.320, .456, .0135, .0106, 2.25, .157, .160)),
    "kidney": _tissue((.273, .483, .0121, .0240, 5.03, .130, .137)),
    "liver": _tissue((.161, .573, .0135, .0238, 4.56, .086, .161)),
    "lung": _tissue((.336, .446, .0215, .0123, 3.91, .212, .168)),
    "muscle_tissue": _tissue((.118, .630, .0100, .0072, 1.53, .064, .059)),
    "pancreas": _tissue((.120, .664, .0403, .0090, 4.56, .060, .060)),
    "skin_tissue": _tissue((.382, .291, .0603, .0044, 1.32, .277, .096)),
    "spleen": _tissue((.207, .579, .0071, .0107, 3.18, .097, .207)),
    "thymus": _tissue((.150, .626, .0168, .0092, 3.18, .075, .075)),
}
# AP for pancreas/thymus was not reported in those tables: liver/spleen proxy.
# Reproductive and rest are aggregate graph nodes with no matched source row:
# muscle composition is the volume-dominant proxy; replace with weighted data.
_RR_ROWS["reproductive"] = _RR_ROWS["muscle_tissue"]
_RR_ROWS["rest"] = _RR_ROWS["muscle_tissue"]

# Plasma and red-cell fractions, Rodgers et al. 2005 Table 1 (dimensionless).
PLASMA_FNL_FRACTION = Distribution(0.0023, 0.0)
PLASMA_FNP_FRACTION = Distribution(0.0013, 0.0)
# Poulin & Theil 2002 (PMID 11977112); plasma water fraction used by the
# modified-Poulin equation (also the existing IVIVE plasma composition).
PLASMA_WATER_FRACTION_UNITLESS = Distribution(0.945, 0.0)
# Poulin & Theil 2002; tissue:plasma albumin ratio in the standard f_ut estimate.
POULIN_ALBUMIN_RATIO_UNITLESS = Distribution(0.5, 0.0)
BC_FIW_FRACTION = Distribution(0.603, 0.0)
BC_FNL_FRACTION = Distribution(0.0017, 0.0)
BC_FNP_FRACTION = Distribution(0.0029, 0.0)
BC_AP_MG_G = Distribution(0.50, 0.0)
# Rodgers et al. 2005: intracellular/blood-cell pH 7.0/7.2; plasma pH 7.4.
IW_PH_UNITLESS = Distribution(7.0, 0.0)
BC_PH_UNITLESS = Distribution(7.2, 0.0)
PLASMA_PH_UNITLESS = Distribution(7.4, 0.0)


def compute_berezhkovskiy_kp(
    logp: float, tissue: RRTissue, fup: float, *, adipose: bool = False
) -> float:
    """Return modified-Poulin tissue:plasma Kp (Berezhkovskiy 2004)."""
    if not 0 < fup <= 1:
        raise ValueError("Berezhkovskiy requires 0 < fup <= 1")

    # GastroPlus Scientific Principles Guide, Eqs. 2-19, 2-21, 2-24:
    # https://gastroplus.slp-software.com/scientific-principles-guide/GPX.2/tissue-distribution-calculations
    p = 10 ** (1.115 * logp - 1.35 if adipose else logp)
    fut = 1 / (1 + ((1 - fup) / fup) * POULIN_ALBUMIN_RATIO_UNITLESS.mean)
    tissue_water = tissue.few.mean + tissue.fiw.mean
    tissue_partition = (
        p * (tissue.fnl.mean + 0.3 * tissue.fnp.mean)
        + 0.7 * tissue.fnp.mean
        + tissue_water / fut
    )
    plasma_partition = (
        p * (PLASMA_FNL_FRACTION.mean + 0.3 * PLASMA_FNP_FRACTION.mean)
        + 0.7 * PLASMA_FNP_FRACTION.mean
        + PLASMA_WATER_FRACTION_UNITLESS.mean / fup
    )
    return tissue_partition / plasma_partition


def compute_rr_kp(
    logp: float,
    pka: float | None,
    compound_type: str,
    tissue: RRTissue,
    fup: float,
    rbp: float,
    hematocrit: float,
    pka_secondary: float | None = None,
    *,
    adipose: bool = False,
) -> float:
    """Return total tissue:plasma Kp from the 2005/2006 Kpu equations."""
    if not 0 < fup <= 1 or rbp <= 0 or not 0 < hematocrit < 1:
        raise ValueError("R&R requires 0 < fup <= 1, rbp > 0, and 0 < hematocrit < 1")

    # Rodgers & Rowland 2006: vegetable-oil partition for adipose.
    p = 10 ** (1.115 * logp - 1.35 if adipose else logp)
    base = compound_type in {"base", "zwitterion"} and pka is not None
    # shortcut: profile carries one basic and one acidic pKa; extend it for polyprotic drugs.
    acid = compound_type in {"acid", "zwitterion"} and (
        pka_secondary if compound_type == "zwitterion" else pka
    ) is not None
    acidic_pka = pka_secondary if compound_type == "zwitterion" else pka

    def ion_terms(ph: float) -> tuple[float, float]:
        b = 10 ** (pka - ph) if base and pka is not None else 0.0
        a = 10 ** (ph - acidic_pka) if acid and acidic_pka is not None else 0.0
        return 1.0 + b + a, b

    x, tissue_base = ion_terms(IW_PH_UNITLESS.mean)
    y, _ = ion_terms(PLASMA_PH_UNITLESS.mean)
    plasma_lipid = (p * PLASMA_FNL_FRACTION.mean
                    + (0.3 * p + 0.7) * PLASMA_FNP_FRACTION.mean) / y
    lipid = (p * tissue.fnl.mean + (0.3 * p + 0.7) * tissue.fnp.mean) / y
    kpu = tissue.few.mean + x * tissue.fiw.mean / y + lipid

    if base and pka is not None and pka >= 7:
        x_bc, bc_base = ion_terms(BC_PH_UNITLESS.mean)
        kpu_bc = (hematocrit - 1.0 + rbp) / (hematocrit * fup)
        bc_lipid = (p * BC_FNL_FRACTION.mean
                    + (0.3 * p + 0.7) * BC_FNP_FRACTION.mean) / y
        # shortcut: floor nonphysical negative affinity; use measured B:P when available.
        ka_ap = max(0.0, (kpu_bc - x_bc * BC_FIW_FRACTION.mean / y - bc_lipid)
                    * y / (BC_AP_MG_G.mean * bc_base))
        kpu += ka_ap * tissue.ap_mg_g.mean * tissue_base / y
    else:
        ratio = (tissue.lipoprotein_ratio if compound_type == "neutral"
                 else tissue.albumin_ratio).mean
        kpu += max(0.0, 1.0 / fup - 1.0 - plasma_lipid) * ratio

    return kpu * fup
