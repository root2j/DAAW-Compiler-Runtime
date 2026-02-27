"""Enumerations used across the DAAW system."""

from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    NEEDS_HUMAN = "needs_human"
    RETRYING = "retrying"


class AgentRole(str, Enum):
    PLANNER = "planner"
    PM = "pm"
    BREAKDOWN = "breakdown"
    CRITIC = "critic"
    USER_PROXY = "user_proxy"
    GENERIC_LLM = "generic_llm"


class PatchAction(str, Enum):
    RETRY = "retry"
    INSERT = "insert"
    REMOVE = "remove"
    UPDATE_INPUT = "update_input"
