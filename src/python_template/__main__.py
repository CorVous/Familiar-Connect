"""Allow running python-template as a module: python -m python_template."""

import sys

from python_template.cli import main

if __name__ == "__main__":
    sys.exit(main())
