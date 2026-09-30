"""Index a repository at a commit. Incremental when an earlier version is already indexed.

  python scripts/build_index.py --repo ./myrepo --commit HEAD
  python scripts/build_index.py --repo ./myrepo --commit abc123 --label commit_002 --base commit_001
"""
import argparse
import json
import logging
import sys

import _bootstrap  # noqa: F401
from aci.config import load_config
from aci.errors import ACIError
from aci.indexing.builder import build_index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", default=None, help="commit/ref (default HEAD; plain directories use the working tree)")
    ap.add_argument("--label", default=None, help="human-friendly version name, e.g. commit_002")
    ap.add_argument("--base", default=None, help="version to update incrementally from (default: latest indexed)")
    ap.add_argument("--full", action="store_true", help="ignore existing versions; rebuild everything (embeddings still come from cache)")
    ap.add_argument("--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        _, stats = build_index(a.repo, a.commit, a.label, load_config(a.config), base=a.base, incremental=not a.full)
    except (ACIError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
