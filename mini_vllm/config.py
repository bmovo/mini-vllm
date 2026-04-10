from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 32_000
    max_position_embeddings: int = 2_048
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2_048
    layer_norm_eps: float = 1e-5
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{self.hidden_size} vs {self.num_attention_heads}"
            )
