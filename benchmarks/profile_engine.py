from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mini_vllm import DecoderOnlyTransformer, MiniLLMEngine, ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile mini continuous batching engine.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--max-position-embeddings", type=int, default=2_048)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=2_048)
    return parser


def validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def main() -> None:
    args = build_parser().parse_args()
    validate_positive("num_requests", args.num_requests)
    validate_positive("prompt_length", args.prompt_length)
    validate_positive("max_new_tokens", args.max_new_tokens)
    validate_positive("max_batch_size", args.max_batch_size)
    validate_positive("steps", args.steps)
    torch.manual_seed(args.seed)

    config = ModelConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
    )
    model = DecoderOnlyTransformer(config)
    engine = MiniLLMEngine(
        model,
        max_batch_size=args.max_batch_size,
        device=args.device,
    )

    prompt_bank = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.num_requests, args.prompt_length),
    ).tolist()
    for index, prompt in enumerate(prompt_bank):
        engine.submit(prompt, max_new_tokens=args.max_new_tokens, request_id=f"req-{index}")

    activities = [ProfilerActivity.CPU]
    sort_key = "self_cpu_time_total"
    if str(args.device).startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
        sort_key = "self_cuda_time_total"

    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.steps):
            if not engine.has_pending_requests():
                break
            with record_function("engine_step"):
                engine.step()

    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.row_limit))


if __name__ == "__main__":
    main()
