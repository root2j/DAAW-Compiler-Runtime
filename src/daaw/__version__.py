"""Single source of truth for the DAAW build version.

Bumped whenever a user-visible change ships so the UI footer / CLI can
confirm which code is actually running (worktree vs. main repo, pip-installed
vs. local, etc.).
"""

from __future__ import annotations

# Semantic version of the DAAW Compiler-Runtime.
__version__ = "0.3.9"

# Short build tag shown in the UI next to the version — bump with features.
BUILD_TAG = "drop-truncated-tasks"
