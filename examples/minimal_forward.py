import torch

from mini_vllm import DecoderOnlyTransformer, ModelConfig


def main() -> None:
    config = ModelConfig(
        vocab_size=1_024,
        max_position_embeddings=128,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
    )
    model = DecoderOnlyTransformer(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids, use_cache=True)

    print(f"logits shape: {tuple(outputs.logits.shape)}")
    print(f"layers in kv cache: {len(outputs.past_key_values or ())}")
    if outputs.past_key_values:
        print(f"per-layer key shape: {tuple(outputs.past_key_values[0][0].shape)}")


if __name__ == "__main__":
    main()
