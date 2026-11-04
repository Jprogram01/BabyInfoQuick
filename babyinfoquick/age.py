from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel


class AgeBand(str, Enum):
    NEWBORN = "newborn"
    INFANT_1_3 = "infant_1_3"
    INFANT_4_6 = "infant_4_6"
    INFANT_7_12 = "infant_7_12"
    TODDLER_12_24 = "toddler_12_24"
    TODDLER_24_36 = "toddler_24_36"


BAND_MONTHS: dict[AgeBand, tuple[float, float]] = {
    AgeBand.NEWBORN: (0.0, 1.0),
    AgeBand.INFANT_1_3: (1.0, 4.0),
    AgeBand.INFANT_4_6: (4.0, 7.0),
    AgeBand.INFANT_7_12: (7.0, 12.0),
    AgeBand.TODDLER_12_24: (12.0, 24.0),
    AgeBand.TODDLER_24_36: (24.0, 36.0),
}

ALL_BANDS: list[AgeBand] = list(BAND_MONTHS)

BAND_LABELS: dict[AgeBand, str] = {
    AgeBand.NEWBORN: "Newborn (0-1 month)",
    AgeBand.INFANT_1_3: "1-3 months",
    AgeBand.INFANT_4_6: "4-6 months",
    AgeBand.INFANT_7_12: "7-12 months",
    AgeBand.TODDLER_12_24: "12-24 months",
    AgeBand.TODDLER_24_36: "24-36 months",
}


class AgeRange(BaseModel):
    min_months: float
    max_months: float

    def bands(self) -> list[AgeBand]:
        out = []
        for band, (lo, hi) in BAND_MONTHS.items():
            if self.min_months < hi and self.max_months > lo:
                out.append(band)
        return out


def bands_for_months(min_months: float, max_months: float) -> list[AgeBand]:
    return AgeRange(min_months=min_months, max_months=max_months).bands()


_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "eighteen": 18,
    "twenty": 20,
    "twenty-four": 24,
}

_UNIT_MONTHS = {
    "day": 1.0 / 30.0,
    "days": 1.0 / 30.0,
    "week": 0.25,
    "weeks": 0.25,
    "month": 1.0,
    "months": 1.0,
    "mo": 1.0,
    "mos": 1.0,
    "year": 12.0,
    "years": 12.0,
    "yr": 12.0,
    "yrs": 12.0,
}

_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_UNIT = r"(" + "|".join(sorted(_UNIT_MONTHS, key=len, reverse=True)) + r")"

_RANGE_RE = re.compile(rf"\b{_NUM}\s*(?:-|–|to)\s*{_NUM}[\s-]*{_UNIT}\b", re.IGNORECASE)
_POINT_RE = re.compile(rf"\b{_NUM}[\s-]*{_UNIT}[\s-]*old\b|\b{_NUM}[\s-]*{_UNIT}\b", re.IGNORECASE)

_KEYWORD_BANDS: list[tuple[re.Pattern[str], list[AgeBand]]] = [
    (re.compile(r"\bnewborn|neonat|just born|days old\b", re.IGNORECASE), [AgeBand.NEWBORN]),
    (
        re.compile(r"\btoddler\b", re.IGNORECASE),
        [AgeBand.TODDLER_12_24, AgeBand.TODDLER_24_36],
    ),
    (
        re.compile(r"\binfant|\bbaby\b", re.IGNORECASE),
        [
            AgeBand.NEWBORN,
            AgeBand.INFANT_1_3,
            AgeBand.INFANT_4_6,
            AgeBand.INFANT_7_12,
        ],
    ),
]


def _to_number(token: str) -> float | None:
    token = token.lower()
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    try:
        return float(token)
    except ValueError:
        return None


def extract_age_bands(text: str) -> list[AgeBand]:
    match = _RANGE_RE.search(text)
    if match:
        lo, hi, unit = match.group(1), match.group(2), match.group(3)
        lo_n, hi_n = _to_number(lo), _to_number(hi)
        if lo_n is not None and hi_n is not None:
            mult = _UNIT_MONTHS[unit.lower()]
            return bands_for_months(lo_n * mult, hi_n * mult)

    for match in _POINT_RE.finditer(text):
        num = match.group(1) or match.group(3)
        unit = match.group(2) or match.group(4)
        value = _to_number(num) if num else None
        if value is None or unit is None:
            continue
        months = value * _UNIT_MONTHS[unit.lower()]
        return bands_for_months(months, months + 0.01)

    for pattern, bands in _KEYWORD_BANDS:
        if pattern.search(text):
            return list(bands)

    return []
