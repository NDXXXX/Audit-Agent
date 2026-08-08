"""Prompt templates used by Audit Agent workflow stages."""

from audit_agent.prompts.stage2 import FINAL_PROMPT
from audit_agent.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from audit_agent.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from audit_agent.prompts.stage5 import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT

__all__ = [
    "FINAL_PROMPT",
    "PLANNER_PROMPT",
    "VERIFIER_PROMPT",
    "CONTEXT_COMPRESSION_PROMPT",
    "CHAT_RESPONDER_PROMPT",
    "INTENT_ROUTER_PROMPT",
]
