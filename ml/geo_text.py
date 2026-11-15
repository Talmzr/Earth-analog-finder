"""Geology text for both bodies + the lithology heuristics used for pseudo-labels and the discrete match term.

Moon text  = USGS Unified Geologic Map unit (name, class, age, description, interpretation) + one composition
             sentence. The map units describe morphology and stratigraphy, not rock type, so the composition
             sentence adds what is established for the whole unit family: mare units are basalt, everything
             else at these sites is feldspathic (anorthositic) highland crust reworked by impacts.
Earth text = tile name + Macrostrat bedrock lithology/description + catalog context (GVP volcano rock type and
             summary, impact-structure record, named-site tags).

Classes are heuristics, not labels: they only define which Moon/Earth text pairs are pulled together or pushed
apart during contrastive fine-tuning, and the small discrete lithology-match term of the geology score.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MARE_UNITS = {"Em", "Im1", "Im2", "Imd"}
DARK_MANTLE_UNITS = {"Id"}
COMP = {
    "mare": "Composition: mare basalt lava flows, iron-rich basaltic rock with pyroxene, plagioclase, olivine and "
            "ilmenite, covered by basaltic regolith.",
    "dark": "Composition: pyroclastic deposits of basaltic volcanic glass beads (dark mantling ash) over basalt.",
    "highland": "Composition: feldspathic highland crust dominated by anorthosite (plagioclase feldspar), with "
                "noritic anorthosite, impact breccia and impact melt, covered by anorthositic regolith.",
}
IMPACT_MOON = re.compile(r"crater|catena|ejecta|secondary|basin|formation", re.I)


# ----------------------------------------------------------------------------- Moon
def moon_family(unit: str, mare_frac: float) -> str:
    if unit in DARK_MANTLE_UNITS:
        return "dark"
    if unit in MARE_UNITS or (mare_frac is not None and np.isfinite(mare_frac) and mare_frac >= 0.5):
        return "mare"
    return "highland"


def moon_classes(unit: str, unit_name: str, mare_frac: float) -> dict:
    fam = moon_family(unit, mare_frac)
    return dict(comp="basaltic" if fam in ("mare", "dark") else "anorthositic",
                impact=bool(IMPACT_MOON.search(str(unit_name))) and unit not in MARE_UNITS | DARK_MANTLE_UNITS,
                family=fam)


def moon_text(unit_row: dict, mare_frac: float) -> str:
    u = unit_row
    parts = [f"{u.get('name', '')} ({u.get('unit_class_name', '')}; {u.get('age', '')} age), lunar geologic map unit."]
    for k in ("description", "interpretation"):
        v = u.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip().rstrip(".") + ".")
    parts.append(COMP[moon_family(u.get("unit", ""), mare_frac)])
    return " ".join(parts)


def moon_texts_for(units_of: pd.Series, mare_frac_of: pd.Series, catalog: pd.DataFrame):
    """Text + classes per row (tile or entity). Rows sharing unit and family share the text."""
    cat = {r["unit"]: r for r in catalog.to_dict("records")}
    texts, classes = [], []
    for unit, mf in zip(units_of, mare_frac_of):
        row = cat.get(unit, dict(unit=unit, name=str(unit), unit_class_name="", age=""))
        texts.append(moon_text(row, mf))
        classes.append(moon_classes(unit, row.get("name", ""), mf))
    return texts, classes


# ----------------------------------------------------------------------------- Earth
def _clean(s, n):
    s = "" if not isinstance(s, str) else re.sub(r"\s+", " ", s).strip()
    return s[:n]


def earth_text(r: dict) -> str:
    """One geology description per Earth tile; '' when nothing geological is known."""
    parts = []
    src = r.get("source", "")
    ctx = _clean(r.get("context"), 700)
    lith, desc, allt = _clean(r.get("mstrat_lith"), 200), _clean(r.get("mstrat_descrip"), 300), _clean(r.get("mstrat_text_all"), 500)
    if src in ("named_site", "impact_structure", "volcano"):
        parts.append(f"{_clean(r.get('name'), 80)}.")
    if lith or allt:
        parts.append(f"Bedrock: {lith or allt}.")
        if desc and desc.lower() != "nan":
            parts.append(desc.rstrip(".") + ".")
        if allt and allt != lith:
            parts.append(allt.rstrip(".") + ".")
    if src in ("named_site", "impact_structure", "volcano") and ctx:
        parts.append(ctx.rstrip(".") + ".")
    elif r.get("stratum") == "ice_sheet":
        parts.append("Ice sheet surface, glacier ice.")
    return " ".join(parts)


RX = {
    "anorth": re.compile(r"anorthosit|troctolit|\bnorit|leucogabbro|layered (mafic |ultramafic )?intrusion"),
    "basalt": re.compile(r"basalt|picrit|hawaiite|tholeiit|basanit|tephrit|mafic volcanic|scoria|cinder cone|dolerite|diabase"),
    "impact": re.compile(r"impact (structure|crater|melt|breccia|site)|impact_crater(?!_like)|impact_melt|suevite|impactite|"
                         r"shatter cone|shock(ed)? metamorph|astrobleme|meteorite crater"),
    "mafic_plut": re.compile(r"gabbro|ultramafic|peridotit|dunite|pyroxenit"),
    "intermediate": re.compile(r"andesit"),
    "silicic": re.compile(r"rhyolit|dacit|trachyt|phonolit|obsidian|pumice|ignimbrite|felsic"),
    "sed": re.compile(r"limestone|dolostone|dolomit|carbonate|evaporite|halite|gypsum|shale|mudstone|sandstone|siltstone|"
                      r"\bclay|chert|\bcoal|marl|chalk|conglomerate|arenite|arkose|greywacke|redbeds|sedimentary|siliciclastic|lakebed"),
    "granmet": re.compile(r"granit|granodiorit|tonalit|diorite|monzo|syenit|gneiss|schist|amphibolit|metamorphic|migmatit|"
                          r"quartzite|slate|phyllite|crystalline"),
    "ice": re.compile(r"ice_sheet|ice sheet|glacier ice"),
    "uncons": re.compile(r"alluvi|colluvi|\btill\b|loess|\bdunes?\b|\bsand\b|gravel|unconsolidated|surficial|regolith"),
    "fragmental": re.compile(r"breccia|talus|scree|tephra|cinder|scoria|volcanic ash|pyroclastic|lapilli"),
    "tube": re.compile(r"lava[ _-]?tubes?|lava caves?|\bcaves?\b"),
}
GVP_BASALTIC = ("basalt", "trachybasalt", "tephrite", "picro")
GVP_INTERMEDIATE = ("andesite", "trachyandesite")
GVP_SILICIC = ("dacite", "rhyolite", "trachyte", "phonolite")


def earth_primary_text(r: dict) -> str:
    """Name, primary rock type and catalog context: where a rock must appear to count as the tile's main lithology."""
    return " ; ".join(str(r.get(k)) for k in ("name", "mstrat_lith", "context") if isinstance(r.get(k), str))


def earth_classes(text: str, source: str = "", ice_frac: float = np.nan, primary: str | None = None) -> dict:
    t = (text or "").lower()
    if not t.strip():
        return dict(known=False, primary="unknown")
    t2 = re.sub(r"basaltic[\s-]+(trachy)?andesite", " andesite ", t)
    c = {k: bool(rx.search(t2 if k == "basalt" else t)) for k, rx in RX.items()}
    m = re.search(r"rock=([^;]+)", t)
    if source == "volcano" and m and "no data" not in m.group(1):
        rock = m.group(1).strip()
        c["basalt"] = rock.startswith(GVP_BASALTIC)                 # catalog rock type beats free text
        c["intermediate"] = rock.startswith(GVP_INTERMEDIATE)
        c["silicic"] = rock.startswith(GVP_SILICIC)
    # anorthosite listed only as a minor component of a long unit description (e.g. inside a gneiss complex)
    # is a partial match, not the tile's lithology
    strong = c["anorth"] and (primary is None or bool(RX["anorth"].search(primary.lower())))
    c["anorth_minor"] = c["anorth"] and not strong
    c["anorth"] = strong
    c["impact"] = c["impact"] or source == "impact_structure"
    c["ice"] = c["ice"] or (np.isfinite(ice_frac) and ice_frac >= 0.7)
    igneous_target = c["anorth"] or c["basalt"] or c["mafic_plut"] or c["granmet"] or c["silicic"] or c["intermediate"]
    c["impact_sed_target"] = c["impact"] and c["sed"] and not igneous_target
    # named analog sites: their curated tags are more reliable than the coarse bedrock map
    c["basalt_strong"] = c["basalt"] and (source in ("volcano", "named_site") or not (c["sed"] or c["granmet"] or c["silicic"]))
    for name in ("anorth", "impact", "basalt_strong", "intermediate", "silicic", "sed", "granmet", "ice", "uncons"):
        if c.get(name):
            c["primary"] = {"basalt_strong": "basalt", "impact": "impact_sed" if c["impact_sed_target"] else "impact"}.get(name, name)
            break
    else:
        c["primary"] = "basalt_weak" if c["basalt"] else ("mafic_plut" if c["mafic_plut"] else "other")
    c["known"] = True
    return c


def relation(mc: dict, ec: dict) -> int:
    """+1 pull together, -1 push apart, 0 leave alone (partial or unknown similarity)."""
    if not ec.get("known"):
        return 0
    comp_match = (mc["comp"] == "basaltic" and ec["basalt_strong"]) or (mc["comp"] == "anorthositic" and ec["anorth"])
    impact_match = mc["impact"] and ec["impact"] and not ec["impact_sed_target"]
    if comp_match or impact_match:
        return 1
    partial = ((mc["impact"] and ec["impact"]) or ec["mafic_plut"] or ec["basalt"] or ec["anorth"] or ec.get("anorth_minor")
               or (mc["comp"] == "basaltic" and ec["intermediate"])
               or (ec["fragmental"] and not (ec["sed"] or ec["granmet"] or ec["ice"]))
               or ec["primary"] in ("uncons", "other"))
    return 0 if partial else -1


def lith_match(mc: dict, ec: dict) -> float:
    """Discrete lithology-match term of the geology score (brief: +0.5 both anorthositic, +0.3 both basaltic;
    +0.2 when both are impact materials in a non-sedimentary target). NaN when the Earth tile has no text."""
    if not ec.get("known"):
        return float("nan")
    s = 0.0
    if mc["comp"] == "anorthositic" and ec["anorth"]:
        s = 0.5
    elif mc["comp"] == "basaltic" and ec["basalt_strong"]:
        s = 0.3
    if mc["impact"] and ec["impact"] and not ec["impact_sed_target"]:
        s += 0.2
    return s
