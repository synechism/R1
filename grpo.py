import gc
import math
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import torch

from data_types import Episode, MiniBatch
from model import Transformer


@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    pad_token_id: int,
    eos_token_id: int,
    max_gen_len: int,
    num_samples_per_prompt: int,
    reward_fn: Callable,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> List[Episode]:
    prompt_token_ids = batch.prompt_token_ids
    num_prompts = len(prompt_token_ids)
    bsz = num_prompts * num_samples_per_prompt

    min_prompt_len = min(len(t) for t in prompt_token_ids)
    max_prompt_len = max(len(t) for t in prompt_token_ids)
    total_len = max_gen_len + max_prompt_len

    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )

    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for i, prompt_ids in enumerate(prompt_token_ids):
        for j in range(num_samples_per_prompt):
            idx = i * num_samples_per_prompt + j
            tokens[idx, :len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    input_text_mask = tokens != pad_token_id
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    prev_pos = 0
    for cur_pos in range(min_prompt_len, total_len):
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = logits.argmax(dim=-1)

        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token

        if eos_token_id is not None:
            is_end = next_token == eos_token_id
            is_generated = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end & is_generated)

        prev_pos = cur_pos

        if is_finished.all():
            break

    model.del_kv_cache()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    episodes = []
    tokens_list = tokens.tolist()
    is_finished_list = is_finished.tolist()

    for i in range(num_prompts):
        prompt_ids = prompt_token_ids[i]
        prompt_len = len(prompt_ids)

        for j in range(num_samples_per_prompt):
            idx = i * num_samples_per_prompt + j

            generated_ids = tokens_list[idx][prompt_len:]
            if pad_token_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(pad_token_id)]

            metadata = batch.metadata[i] if batch.metadata else None
            reward_result = reward_fn(
                prompt=batch.prompts[i],
                generated_ids=generated_ids,
                metadata=metadata,
            )

            episode = Episode(
                prompt=batch.prompts[i],
                prompt_token_ids=prompt_ids,
                generated_text=reward_result.get("generated_text", ""),
                generated_token_ids=generated_ids,
                full_text=batch.prompts[i] + reward_result.get("generated_text", ""),
                is_finished=is_finished_list[idx],
                reward=reward_result["reward"],
                reward_info=reward_result.get("reward_info", {}),
            )
            episodes.append(episode)

    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    from dataclasses import replace

    groups = defaultdict(list)
    for episode in episodes:
        key = tuple(episode.prompt_token_ids)
        groups[key].append(episode)

    output = []
    for group in groups.values():
        rewards = [ep.reward for ep in group]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        for episode in group:
            advantage = (episode.reward - mean_reward) / (std_reward + 1e-8)
            output.append(replace(episode, reward=advantage))

    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def update_policy(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prompt_token_ids) + len(x.generated_token_ids))

    num_target_tokens = sum(len(ep.generated_token_ids) for ep in episodes)
    total_entropy = 0.0
    total_loss = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]

        batch_lengths = [
            len(ep.prompt_token_ids) + len(ep.generated_token_ids)
            for ep in batch_episodes
        ]
        max_len = max(batch_lengths)

        batch_token_ids = []
        batch_masks = []
        batch_advantages = []

        for k, ep in enumerate(batch_episodes):
            full_ids = ep.prompt_token_ids + ep.generated_token_ids
            padding = [pad_token_id] * (max_len - len(full_ids))

            batch_token_ids.append(full_ids + padding)

            mask = [0] * len(ep.prompt_token_ids) + [1] * len(ep.generated_token_ids) + [0] * len(padding)
            batch_masks.append(mask)

            batch_advantages.append(ep.reward)

        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_ids = batch_token_ids[:, :-1]
            target_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]

            logits = model(input_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            total_entropy += (token_entropy * target_masks).sum().item() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens

        loss = -obj
        loss.backward()
        total_loss += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "entropy": total_entropy,
    }
