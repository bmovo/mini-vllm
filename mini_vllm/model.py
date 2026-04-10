from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig

PastKeyValue = tuple[Tensor, Tensor]


@dataclass(slots=True)
class DecoderOnlyTransformerOutput:
    logits: Tensor
    hidden_states: Tensor
    past_key_values: tuple[PastKeyValue, ...] | None = None


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.activation = nn.GELU(approximate="tanh")
        self.fc_out = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, PastKeyValue | None]:
        batch_size, query_length, hidden_size = hidden_states.shape

        query = self._reshape_heads(self.q_proj(hidden_states), batch_size, query_length)
        key = self._reshape_heads(self.k_proj(hidden_states), batch_size, query_length)
        value = self._reshape_heads(self.v_proj(hidden_states), batch_size, query_length)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.size(-2)
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        present = (key, value) if use_cache else None
        key_length = key.size(-2)

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        causal_mask = self._build_causal_mask(
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            device=hidden_states.device,
        )
        attn_scores = attn_scores.masked_fill(
            ~causal_mask, torch.finfo(attn_scores.dtype).min
        )

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.size(1) != key_length:
                raise ValueError(
                    "attention_mask must have shape [batch_size, total_kv_length]: "
                    f"got {tuple(attention_mask.shape)} for key length {key_length}"
                )
            expanded_mask = attention_mask[:, None, None, :].to(torch.bool)
            attn_scores = attn_scores.masked_fill(
                ~expanded_mask, torch.finfo(attn_scores.dtype).min
            )

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_length, hidden_size
        )
        attn_output = self.out_proj(attn_output)
        return self.resid_dropout(attn_output), present

    def _reshape_heads(self, tensor: Tensor, batch_size: int, seq_len: int) -> Tensor:
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _build_causal_mask(
        self,
        query_length: int,
        key_length: int,
        past_length: int,
        device: torch.device,
    ) -> Tensor:
        query_positions = past_length + torch.arange(query_length, device=device)[:, None]
        key_positions = torch.arange(key_length, device=device)[None, :]
        return (key_positions <= query_positions).unsqueeze(0).unsqueeze(0)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, PastKeyValue | None]:
        attn_input = self.input_layernorm(hidden_states)
        attn_output, present = self.self_attn(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output

        mlp_input = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(mlp_input)
        hidden_states = hidden_states + mlp_output
        return hidden_states, present


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool = False,
    ) -> DecoderOnlyTransformerOutput:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape [batch_size, seq_len], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.layers):
                raise ValueError(
                    "past_key_values length must match num_hidden_layers: "
                    f"{len(past_key_values)} vs {len(self.layers)}"
                )
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = (
                torch.arange(
                    past_length, past_length + seq_len, device=input_ids.device, dtype=torch.long
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same shape as input_ids: "
                f"{tuple(position_ids.shape)} vs {tuple(input_ids.shape)}"
            )

        max_position = int(position_ids.max().item()) + 1
        if max_position > self.config.max_position_embeddings:
            raise ValueError(
                "position_ids exceed max_position_embeddings: "
                f"{max_position} > {self.config.max_position_embeddings}"
            )

        total_kv_length = past_length + seq_len
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                total_kv_length,
                dtype=torch.bool,
                device=input_ids.device,
            )
        else:
            attention_mask = attention_mask.to(torch.bool)
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"attention_mask must have shape [batch_size, seq_len], got {tuple(attention_mask.shape)}"
                )
            if attention_mask.size(0) != batch_size:
                raise ValueError(
                    "attention_mask batch dimension must match input_ids: "
                    f"{attention_mask.size(0)} vs {batch_size}"
                )
            if attention_mask.size(1) == seq_len and past_length > 0:
                prefix = torch.ones(
                    batch_size,
                    past_length,
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([prefix, attention_mask], dim=-1)
            elif attention_mask.size(1) != total_kv_length:
                raise ValueError(
                    "attention_mask length must match current sequence or total kv length: "
                    f"{attention_mask.size(1)} vs {seq_len}/{total_kv_length}"
                )

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        hidden_states = self.dropout(hidden_states)

        next_past_key_values: list[PastKeyValue] | None = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            layer_past = past_key_values[layer_index] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present is not None:
                next_past_key_values.append(present)

        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        return DecoderOnlyTransformerOutput(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=tuple(next_past_key_values) if next_past_key_values is not None else None,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
