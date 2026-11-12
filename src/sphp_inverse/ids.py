"""Run IDs for COMSOL jobs: a number that fits a 32-bit signed integer (COMSOL parameter `ID`).

Format YYMMDDTSSS: YY = years after 2020, MMDD = date, T = category (0-9), SSS = sequence (0-999).
Same scheme as the original datagen helper.encode_id, e.g. 2026-02-17, category 1, sequence 0 -> 602171000.
"""
from __future__ import annotations

from datetime import date

BASE_YEAR = 2020
INT32_MAX = 2_147_483_647


def encode_id(category: int, sequence: int, day: date | None = None) -> int:
    day = day or date.today()
    if not 0 <= category <= 9:
        raise ValueError("category must be 0-9")
    if not 0 <= sequence <= 999:
        raise ValueError("sequence must be 0-999")
    yy = day.year - BASE_YEAR
    if yy < 0:
        raise ValueError("date is before the base year")
    run_id = int(f"{yy:02d}{day:%m%d}{category:1d}{sequence:03d}")
    if run_id > INT32_MAX:
        raise OverflowError("ID exceeds the 32-bit integer limit")
    return run_id


def decode_id(run_id: int) -> dict:
    s = str(run_id).zfill(10)
    return {"year": int(s[:2]) + BASE_YEAR, "month": int(s[2:4]), "day": int(s[4:6]),
            "category": int(s[6]), "sequence": int(s[7:])}
