from dataclasses import dataclass


@dataclass
class AgentResponse:
    stop_reason: str
    content: list[dict]
    usage: dict
