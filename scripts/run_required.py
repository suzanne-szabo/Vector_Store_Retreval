"""Run the nine required queries (specs/agent.md §7); the logic lives in rag_agent.required.

    uv run python scripts/run_required.py [--only Q3 Q8] [--out-dir runs/<name>]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the project is not an installed package

from rag_agent.required import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
