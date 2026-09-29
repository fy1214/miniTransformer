"""Module entry point: ``python -m logpdiff <command> ...`` (see cli.py)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
