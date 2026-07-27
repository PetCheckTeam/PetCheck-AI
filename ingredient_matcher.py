# ingredient_matcher.py

import csv
import re
from pathlib import Path


def normalize_name(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return text

def open_csv_with_fallback(csv_path: str | Path):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    last_error = None

    for encoding in encodings:
        try:
            file = open(csv_path, "r", encoding=encoding, newline="")
            file.read(1)
            file.seek(0)
            return file
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error
def load_alias_map(csv_path: str | Path) -> dict[str, dict]:
    alias_map = {}

    with open_csv_with_fallback(csv_path) as file:
        reader = csv.DictReader(file)

        for row in reader:
            alias = row["aliasName"]
            normalized_alias = normalize_name(alias)

            alias_map[normalized_alias] = {
                "ingredientId": row["ingredientId"],
                "standardName": row["standardName"],
                "aliasName": alias,
            }

    return alias_map


def match_ingredients(
    raw_ingredients: list[str],
    alias_map: dict[str, dict],
) -> list[dict]:
    results = []

    for raw_name in raw_ingredients:
        normalized = normalize_name(raw_name)
        matched = alias_map.get(normalized)

        if matched:
            results.append({
                "rawName": raw_name,
                "matchedIngredients": [
                    {
                        "ingredientId": matched["ingredientId"],
                        "standardName": matched["standardName"],
                    }
                ],
                "matchType": "ALIAS",
                "confidence": 1.0,
            })
        else:
            results.append({
                "rawName": raw_name,
                "matchedIngredients": [],
                "matchType": "UNKNOWN",
                "confidence": 0.0,
            })

    return results
