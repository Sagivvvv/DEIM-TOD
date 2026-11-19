"""Run or resume the TinyPerson official 786-image postprocessing comparison."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from official_tile_candidates import CACHE, GT, IMAGE_COUNT, OUTPUT, REPORTS, SPLIT, run_dir, sha256
from report_tile_postprocess import METHODS


TOOLS = Path(__file__).resolve().parent
def complete(runs, method, cache_sha):
    marker = runs / f"{method}.json"
    if not marker.is_file():
        return False
    try:
        result = json.loads(marker.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        return False
    return (
        result.get("method") == method
        and result.get("split") == SPLIT
        and result.get("images") == IMAGE_COUNT
        and result.get("inputs", {}).get("cache_sha256") == cache_sha
        and all((runs / f"{method}.{kind}.json.gz").is_file()
                for kind in ("predictions", "timing", "ap"))
    )


def run(script, *args):
    subprocess.run([sys.executable, "-B", "-X", "utf8", TOOLS / script, *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(METHODS), help="Run one method and its NMS reference.")
    parser.add_argument("--list", action="store_true", help="Show completed and pending methods, then exit.")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--gt", type=Path, default=GT)
    args = parser.parse_args()
    if args.cache.suffix.lower() != ".npz":
        parser.error("Use the new pre-NMS .npz export")
    if not args.cache.is_file():
        parser.error(f"Candidate cache missing: {args.cache}. Run export_official_786_candidates.py first.")
    cache_sha = sha256(args.cache)
    runs = run_dir(OUTPUT, cache_sha)
    report = REPORTS / runs.name
    if args.list:
        for method, (name, _) in METHODS.items():
            status = "done" if complete(runs, method, cache_sha) else "pending"
            print(f"{status:7} {method:17} {name}")
        return

    methods = list(METHODS) if args.method is None else list(dict.fromkeys(["nms", args.method]))
    for method in methods:
        if complete(runs, method, cache_sha):
            print(f"Already completed: {method}", flush=True)
            continue
        print(f"Running: {method}", flush=True)
        run("benchmark_tile_postprocess.py", "--method", method, "--gt", args.gt, "--cache", args.cache)
    run("report_tile_postprocess.py", "--runs", runs, "--gt", args.gt, "--output", report)
    print(report / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
