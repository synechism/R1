import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_types import Episode, MiniBatch
from grpo import rollout, update_policy
from model import ModelArgs, Transformer
from reward import create_math_reward_fn


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 102400):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self._tokenizer = None

    def encode(self, text: str) -> List[int]:
        if self._tokenizer:
            return self._tokenizer.encode(text)
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if self._tokenizer:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t > 2]
        return "".join(chr(t) if 32 <= t < 127 else "?" for t in token_ids)

    @classmethod
    def from_pretrained(cls, path: str):
        tokenizer = cls()
        try:
            from transformers import AutoTokenizer
            tokenizer._tokenizer = AutoTokenizer.from_pretrained(path)
            tokenizer.pad_token_id = tokenizer._tokenizer.pad_token_id or 0
            tokenizer.eos_token_id = tokenizer._tokenizer.eos_token_id or 1
            tokenizer.bos_token_id = tokenizer._tokenizer.bos_token_id or 2
            tokenizer.vocab_size = tokenizer._tokenizer.vocab_size
        except ImportError:
            print("Warning: transformers not installed, using placeholder tokenizer")
        return tokenizer


class MathReasoningDataset(Dataset):
    SYSTEM_PROMPT = (
        "You are a helpful assistant that solves math problems step by step. "
        "Show your reasoning in <think></think> tags, then provide your final answer "
        "in <answer></answer> tags."
    )

    PROMPT_TEMPLATE = "{system}\n\nProblem: {problem}\n\nSolution: <think>"

    def __init__(
        self,
        problems: List[Dict],
        tokenizer: SimpleTokenizer,
    ):
        self.problems = problems
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx) -> Dict:
        item = self.problems[idx]

        prompt = self.PROMPT_TEMPLATE.format(
            system=self.SYSTEM_PROMPT,
            problem=item["problem"],
        )

        return {
            "prompt": prompt,
            "prompt_token_ids": self.tokenizer.encode(prompt),
            "metadata": {
                "expected_answer": item.get("answer"),
                "answer_type": item.get("answer_type", "math"),
            },
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> MiniBatch:
        return MiniBatch(
            prompts=[item["prompt"] for item in batch],
            prompt_token_ids=[item["prompt_token_ids"] for item in batch],
            metadata=[item["metadata"] for item in batch],
        )


def create_dummy_dataset(tokenizer: SimpleTokenizer, num_samples: int = 100) -> MathReasoningDataset:
    import random

    problems = []
    for i in range(num_samples):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            answer = a + b
        elif op == "-":
            answer = a - b
        else:
            answer = a * b

        problems.append({
            "problem": f"What is {a} {op} {b}?",
            "answer": answer,
            "answer_type": "math",
        })

    return MathReasoningDataset(problems, tokenizer)


def train(config: Dict):
    device = torch.device(config["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["dtype"], torch.float32)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print(f"Device: {device}, dtype: {dtype}")

    tokenizer = SimpleTokenizer(vocab_size=config["vocab_size"])

    model_args = ModelArgs(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_dense_layers=config["n_dense_layers"],
        n_routed_experts=config["n_routed_experts"],
        n_activated_experts=config["n_activated_experts"],
        n_shared_experts=config["n_shared_experts"],
        moe_inter_dim=config["moe_inter_dim"],
        inter_dim=config["inter_dim"],
        max_seq_len=config["max_seq_len"],
        kv_lora_rank=config["kv_lora_rank"],
        qk_nope_head_dim=config["qk_nope_head_dim"],
        qk_rope_head_dim=config["qk_rope_head_dim"],
        v_head_dim=config["v_head_dim"],
    )

    print(f"Initializing model with config: {model_args}")
    model = Transformer(model_args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]),
    )

    dataset = create_dummy_dataset(tokenizer, num_samples=config["num_train_samples"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["num_prompts_per_batch"],
        shuffle=True,
        collate_fn=MathReasoningDataset.collate_fn,
    )

    reward_fn = create_math_reward_fn(tokenizer)

    print("\nStarting GRPO training...")
    print("=" * 60)

    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        for batch_idx, batch in enumerate(dataloader):
            global_step += 1

            model.eval()
            episodes = rollout(
                model=model,
                batch=batch,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_gen_len=config["max_gen_len"],
                num_samples_per_prompt=config["num_samples_per_prompt"],
                reward_fn=reward_fn,
                device=device,
                dtype=dtype,
                temperature=config["temperature"],
            )

            if config["skip_unfinished"]:
                episodes = [ep for ep in episodes if ep.is_finished]

            if not episodes:
                print(f"Step {global_step}: No valid episodes, skipping...")
                continue

            model.train()
            results = update_policy(
                model=model,
                optimizer=optimizer,
                episodes=episodes,
                micro_batch_size=config["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=config["max_grad_norm"],
                device=device,
                dtype=dtype,
            )

            rewards = [ep.reward for ep in episodes]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            num_finished = sum(1 for ep in episodes if ep.is_finished)
            mean_gen_len = np.mean([len(ep.generated_token_ids) for ep in episodes])

            elapsed = time.time() - start_time

            print(
                f"Step {global_step:4d} | "
                f"Loss: {results['loss']:.4f} | "
                f"Reward: {mean_reward:.3f} (±{std_reward:.3f}) | "
                f"Finished: {num_finished}/{len(episodes)} | "
                f"GenLen: {mean_gen_len:.1f} | "
                f"GradNorm: {results['grad_norm']:.2f} | "
                f"Time: {elapsed:.1f}s"
            )

            if global_step % config["save_interval"] == 0:
                ckpt_path = ckpt_dir / f"checkpoint_{global_step:06d}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "config": config,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nTraining complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for DeepSeek-R1")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-dense-layers", type=int, default=1)
    parser.add_argument("--n-routed-experts", type=int, default=8)
    parser.add_argument("--n-activated-experts", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=1)
    parser.add_argument("--moe-inter-dim", type=int, default=512)
    parser.add_argument("--inter-dim", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--kv-lora-rank", type=int, default=64)
    parser.add_argument("--qk-nope-head-dim", type=int, default=32)
    parser.add_argument("--qk-rope-head-dim", type=int, default=32)
    parser.add_argument("--v-head-dim", type=int, default=32)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-train-samples", type=int, default=100)
    parser.add_argument("--num-prompts-per-batch", type=int, default=4)
    parser.add_argument("--num-samples-per-prompt", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--skip-unfinished", action="store_true")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "vocab_size": args.vocab_size,
            "dim": args.dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "n_dense_layers": args.n_dense_layers,
            "n_routed_experts": args.n_routed_experts,
            "n_activated_experts": args.n_activated_experts,
            "n_shared_experts": args.n_shared_experts,
            "moe_inter_dim": args.moe_inter_dim,
            "inter_dim": args.inter_dim,
            "max_seq_len": args.max_seq_len,
            "kv_lora_rank": args.kv_lora_rank,
            "qk_nope_head_dim": args.qk_nope_head_dim,
            "qk_rope_head_dim": args.qk_rope_head_dim,
            "v_head_dim": args.v_head_dim,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "num_epochs": args.num_epochs,
            "num_train_samples": args.num_train_samples,
            "num_prompts_per_batch": args.num_prompts_per_batch,
            "num_samples_per_prompt": args.num_samples_per_prompt,
            "micro_batch_size": args.micro_batch_size,
            "max_gen_len": args.max_gen_len,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": args.betas,
            "max_grad_norm": args.max_grad_norm,
            "skip_unfinished": args.skip_unfinished,
            "save_interval": args.save_interval,
            "ckpt_dir": args.ckpt_dir,
        }

    train(config)


if __name__ == "__main__":
    main()
