"""Allow running familiar-connect as a module: python -m familiar_connect."""

import sys

from familiar_connect.cli import main

if __name__ == "__main__":
    sys.exit(main())
