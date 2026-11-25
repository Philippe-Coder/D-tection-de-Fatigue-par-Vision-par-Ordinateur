"""Lance Fatigue Guard sans installation : python main.py [gui|run|stats|…]."""
import sys

from fatigue_guard.cli import main

if __name__ == "__main__":
    sys.exit(main())
