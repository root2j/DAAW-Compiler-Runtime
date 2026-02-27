"""Critic — evaluates task outputs and proposes workflow patches."""

from daaw.critic.critic import Critic
from daaw.critic.patch import apply_patch

__all__ = ["Critic", "apply_patch"]
