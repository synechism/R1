from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Episode:
    prompt: str
    prompt_token_ids: List[int]
    generated_text: str
    generated_token_ids: List[int]
    full_text: str
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]


@dataclass
class MiniBatch:
    prompts: List[str]
    prompt_token_ids: List[List[int]]
    metadata: Optional[List[Dict]] = None
