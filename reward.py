import re
from typing import Any, Callable, Dict, List, Optional

import torch


class RewardModel:
    def __call__(
        self,
        prompt: str,
        generated_ids: List[int],
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RuleBasedRewardModel(RewardModel):
    def __init__(self, tokenizer, format_weight: float = 0.1):
        self.tokenizer = tokenizer
        self.format_weight = format_weight

    def __call__(
        self,
        prompt: str,
        generated_ids: List[int],
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        generated_text = self.tokenizer.decode(generated_ids)

        format_reward = self._compute_format_reward(generated_text)

        answer_reward = 0.0
        if metadata and "expected_answer" in metadata:
            answer_reward = self._compute_answer_reward(
                generated_text,
                metadata["expected_answer"],
                metadata.get("answer_type", "text"),
            )

        total_reward = self.format_weight * format_reward + answer_reward

        return {
            "reward": total_reward,
            "generated_text": generated_text,
            "reward_info": {
                "format_reward": format_reward,
                "answer_reward": answer_reward,
            },
        }

    def _compute_format_reward(self, text: str) -> float:
        think_pattern = r"<think>.*?</think>"
        answer_pattern = r"<answer>.*?</answer>"
        full_pattern = r"^.*<think>.*?</think>\s*<answer>.*?</answer>.*$"

        has_think = bool(re.search(think_pattern, text, re.DOTALL))
        has_answer = bool(re.search(answer_pattern, text, re.DOTALL))
        full_match = bool(re.match(full_pattern, text, re.DOTALL))

        if full_match:
            return 1.0
        reward = 0.0
        if has_think:
            reward += 0.3
        if has_answer:
            reward += 0.5
        return reward

    def _compute_answer_reward(self, text: str, expected: Any, answer_type: str) -> float:
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not answer_match:
            return 0.0

        answer_content = answer_match.group(1).strip()

        if answer_type == "math":
            try:
                result = eval(answer_content, {"__builtins__": {}}, {})
                if abs(float(result) - float(expected)) < 1e-5:
                    return 1.0
            except:
                pass
            return 0.0

        elif answer_type == "choice":
            answer_letter = answer_content.strip().upper()
            if len(answer_letter) == 1 and answer_letter == expected.upper():
                return 1.0
            return 0.0

        else:
            if answer_content.lower().strip() == str(expected).lower().strip():
                return 1.0
            return 0.0


class HuggingFaceRewardModel(RewardModel):
    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        device: str = "cuda",
        tokenizer=None,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        self.rm_model.eval()

        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        generated_ids: List[int],
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        generated_text = self.tokenizer.decode(generated_ids) if self.tokenizer else ""
        full_text = prompt + generated_text

        inputs = self.rm_tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        outputs = self.rm_model(**inputs)
        reward = outputs.logits.squeeze().item()

        return {
            "reward": reward,
            "generated_text": generated_text,
            "reward_info": {"raw_score": reward},
        }


class CompositeRewardModel(RewardModel):
    def __init__(self, reward_models: List[tuple], tokenizer=None):
        self.reward_models = reward_models
        self.tokenizer = tokenizer

    def __call__(
        self,
        prompt: str,
        generated_ids: List[int],
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        total_reward = 0.0
        reward_info = {}
        generated_text = ""

        for weight, rm in self.reward_models:
            result = rm(prompt, generated_ids, metadata)
            total_reward += weight * result["reward"]
            reward_info[rm.__class__.__name__] = result["reward_info"]
            if not generated_text and result.get("generated_text"):
                generated_text = result["generated_text"]

        return {
            "reward": total_reward,
            "generated_text": generated_text,
            "reward_info": reward_info,
        }


def create_math_reward_fn(tokenizer) -> Callable:
    def reward_fn(
        prompt: str,
        generated_ids: List[int],
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        format_reward = 0.0
        if "<think>" in generated_text and "</think>" in generated_text:
            format_reward += 0.3
        if "<answer>" in generated_text and "</answer>" in generated_text:
            format_reward += 0.5
            if re.search(r"</think>\s*<answer>", generated_text):
                format_reward += 0.2

        answer_reward = 0.0
        if metadata and "expected_answer" in metadata:
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                expected = metadata["expected_answer"]

                try:
                    if abs(float(eval(answer_content)) - float(expected)) < 1e-5:
                        answer_reward = 1.0
                except:
                    if answer_content.strip() == str(expected).strip():
                        answer_reward = 1.0

        total_reward = 0.1 * format_reward + answer_reward

        return {
            "reward": total_reward,
            "generated_text": generated_text,
            "reward_info": {
                "format_reward": format_reward,
                "answer_reward": answer_reward,
            },
        }

    return reward_fn
