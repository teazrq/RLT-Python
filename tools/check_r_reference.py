"""Verify the source pinned for live R comparisons (standard library only)."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", nargs="?", type=Path)
    parser.add_argument("--revision", action="store_true")
    args = parser.parse_args()
    reference = json.loads(Path(__file__).with_name("r-reference.json").read_text())
    if args.revision:
        print(reference["revision"])
        return
    if args.checkout is None:
        parser.error("supply an R checkout or --revision")
    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != reference["revision"]:
        raise SystemExit(f"R revision differs: expected {reference['revision']}, found {revision}")
    for name, expected in reference["source_sha256_lf"].items():
        content = (args.checkout / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(content).hexdigest() != expected:
            raise SystemExit(f"R reference content differs: {name}")
    print(f"Verified R revision {revision} and {len(reference['source_sha256_lf'])} source files")


if __name__ == "__main__":
    main()
