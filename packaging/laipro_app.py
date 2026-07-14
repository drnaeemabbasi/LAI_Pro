"""
Entry point for the bundled desktop app.

When launched with no arguments (double-clicked exe), it opens the napari GUI.
It still accepts the normal CLI subcommands if run from a console.
"""
import sys
from laipro.cli import main

if __name__ == "__main__":
    main(sys.argv[1:] or ["gui"])
