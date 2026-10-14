"""`python -m egolearn.data.packed {pack,verify,info}`: build, check or describe
a packed frame store (see the package docstring)."""

import sys

from egolearn.data.packed import main

if __name__ == "__main__":
    sys.exit(main())
