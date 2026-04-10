from __future__ import annotations

import argparse
import statistics
import time

import torch

from mini_vllm import DecoderOnlyTransformer, MiniLLMEngine, ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark mini continuous batching engine.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--arrival-batch-size", type=int, default=8)
    parser.add_argument("--arrival-stride-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--max-position-embeddings", type=int, default=2_048)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=2_048)
    return parser


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def main() -> None:
    args = build_parser().parse_args()
    validate_positive("num_requests", args.num_requests)
    validate_positive("prompt_length", args.prompt_length)
    validate_positive("max_new_tokens", args.max_new_tokens)
    validate_positive("max_batch_size", args.max_batch_size)
    validate_positive("arrival_batch_size", args.arrival_batch_size)
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

    next_request_index = 0
    step_index = 0
    benchmark_start = time.perf_counter()

    while next_request_index < args.num_requests or engine.has_pending_requests():
        should_submit = (
            next_request_index < args.num_requests
            and step_index % max(1, args.arrival_stride_steps) == 0
        )
        if should_submit:
            for _ in range(args.arrival_batch_size):
                if next_request_index >= args.num_requests:
                    break
                engine.submit(
                    prompt_bank[next_request_index],
                    max_new_tokens=args.max_new_tokens,
                    request_id=f"req-{next_request_index}",
                )
                next_request_index += 1

        engine.step()
        step_index += 1

    benchmark_end = time.perf_counter()
    results = engine.finished_results()

    total_prompt_tokens = sum(len(result.prompt_token_ids) for result in results)
    total_output_tokens = sum(len(result.output_token_ids) for result in results)
    ttft_values = [value for value in (result.ttft_seconds for result in results) if value is not None]
    latency_values = [result.latency_seconds for result in results]
    elapsed = benchmark_end - benchmark_start

    print(f"requests: {len(results)}")
    print(f"elapsed_seconds: {elapsed:.4f}")
    print(f"prompt_tokens: {total_prompt_tokens}")
    print(f"generated_tokens: {total_output_tokens}")
    print(f"prompt_throughput_tok_per_s: {total_prompt_tokens / elapsed:.2f}")
    print(f"decode_throughput_tok_per_s: {total_output_tokens / elapsed:.2f}")
    print(f"end_to_end_tok_per_s: {(total_prompt_tokens + total_output_tokens) / elapsed:.2f}")
    print(f"ttft_p50_ms: {statistics.median(ttft_values) * 1000:.2f}")
    print(f"ttft_p95_ms: {percentile(ttft_values, 0.95) * 1000:.2f}")
    print(f"latency_p50_ms: {statistics.median(latency_values) * 1000:.2f}")
    print(f"latency_p95_ms: {percentile(latency_values, 0.95) * 1000:.2f}")


if __name__ == "__main__":
    main()
