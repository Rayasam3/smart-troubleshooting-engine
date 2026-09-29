"""Audit the reference samples against schema.py + the rulebook.
Usage: python -m scripts.validate_samples [--strict]
"""
import json
import sys

from app.config import SAMPLES_DIR
from app.data_loader import catalog_uris
from app.pipeline.validators import audit_response


def main(strict: bool = False) -> int:
    worst = 0
    for path in sorted(SAMPLES_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = audit_response(payload["response"], set(catalog_uris()))
        errors = [i for i in report["issues"] if i["severity"] == "error"]
        status = "PASS" if report["schema_valid"] and not errors else "RULE VIOLATIONS"
        print(f"\n{path.name}: schema_valid={report['schema_valid']} errors={len(errors)} -> {status}")
        for e in report["schema_errors"]:
            print(f"   [schema] {e}")
        for i in report["issues"]:
            print(f"   [{i['severity']}] {i['code']:<24} {i['path']}: {i['message']}")
        if not report["schema_valid"] or errors:
            worst = 1
    return worst if strict else 0


if __name__ == "__main__":
    sys.exit(main(strict="--strict" in sys.argv))