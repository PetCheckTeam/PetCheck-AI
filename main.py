# main.py
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

from ingredient_extractor import extract_ingredients
from ingredient_matcher import load_alias_map, match_ingredients


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parent
ALIAS_CSV_PATH = BASE_DIR / "data" / "petcheck_ingredient_aliases.csv"

alias_map = load_alias_map(ALIAS_CSV_PATH)


def analyze_ocr_text(ocr_text: str) -> dict:
    extraction_result = extract_ingredients(ocr_text)

    raw_ingredients = extraction_result["ingredients"]

    matched_ingredients = match_ingredients(
        raw_ingredients=raw_ingredients,
        alias_map=alias_map,
    )

    return {
        "extraction": extraction_result,
        "matchedIngredients": matched_ingredients,
    }


if __name__ == "__main__":
    sample_path = BASE_DIR / "sample_ocr.txt"

    with open(sample_path, "r", encoding="utf-8-sig") as file:
        ocr_text = file.read()

    result = analyze_ocr_text(ocr_text)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    """확인"""
