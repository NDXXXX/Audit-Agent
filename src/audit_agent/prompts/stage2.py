"""Prompts for the Audit Agent's final report template."""

FINAL_PROMPT = """Audit Agent review report

Status: {status}
Task: {task}
Review cycles: {attempts}/{max_attempts}
Review summary: {last_actor_summary}
Verification: {verification_reason}
Next actions: {recommended_next_instruction}
"""
