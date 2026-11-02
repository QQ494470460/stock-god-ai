"""Repo-root conftest: makes ``src`` / ``config`` importable from anywhere.

pytest's ``pythonpath = .`` in ``pytest.ini`` already handles the normal case;
this is a belt-and-suspenders fallback for invocation styles that skip the ini
file. Side-effect free: no network, no filesystem writes.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
