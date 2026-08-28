from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentage(covered: int, total: int) -> float:
    return 100.0 if total == 0 else covered * 100.0 / total


def main() -> int:
    parser = argparse.ArgumentParser(description="Check separate coverage thresholds.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--statements", type=float, default=85.0)
    parser.add_argument("--branches", type=float, default=75.0)
    args = parser.parse_args()

    totals = json.loads(args.report.read_text(encoding="utf-8"))["totals"]
    statement_rate = percentage(totals["covered_lines"], totals["num_statements"])
    branch_rate = percentage(totals["covered_branches"], totals["num_branches"])
    print(f"statement coverage: {statement_rate:.2f}% (required {args.statements:.2f}%)")
    print(f"branch coverage: {branch_rate:.2f}% (required {args.branches:.2f}%)")
    return int(statement_rate < args.statements or branch_rate < args.branches)


if __name__ == "__main__":
    raise SystemExit(main())
