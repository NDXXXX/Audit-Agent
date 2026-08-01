"""Prompt templates used by DDclaw workflow stages."""

from ddclaw.prompts.stage2 import (
    ACTOR_PROMPT,
    FINAL_PROMPT,
)
from ddclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from ddclaw.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from ddclaw.prompts.stage5 import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT

__all__ = [
    "ACTOR_PROMPT",
    "FINAL_PROMPT",
    "PLANNER_PROMPT",
    "VERIFIER_PROMPT",
    "CONTEXT_COMPRESSION_PROMPT",
    "CHAT_RESPONDER_PROMPT",
    "INTENT_ROUTER_PROMPT",
]
