"""Same seed and optimizer budget per condition, including Table 2's structure switches."""
import json
from pathlib import Path
from train import args_from, run

suite = [
    ("backbone", "none"), ("hcen", "none"), ("hcen-aul", "none"),
    ("full", "none"), ("full", "ratr"), ("full", "cov"), ("full", "cov_consistency"),
]
rows = []
for structure, variant in suite:
    result = run(args_from(["--structure", structure, "--variant", variant]))
    rows.append(result)
    assert result["steps"] == 2
    assert result["evaluation"]["backbone"]["valid_queries"] == 6
    print(structure, variant, "OK", flush=True)
Path("smoke-results.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n",
                                      encoding="utf-8")
