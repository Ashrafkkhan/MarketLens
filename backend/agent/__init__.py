"""
backend/agent — MarketLens Phase 5: GenAI Orchestration Agent
"""
from backend.agent.prompt_templates import SYNTHESIS_SYSTEM_PROMPT, format_synthesis_prompt
from backend.agent.synthesizer import ReportSynthesizer, get_synthesizer
from backend.agent.orchestrator import MarketLensAgent, get_agent

__all__ = [
    "SYNTHESIS_SYSTEM_PROMPT",
    "format_synthesis_prompt",
    "ReportSynthesizer",
    "get_synthesizer",
    "MarketLensAgent",
    "get_agent",
]
