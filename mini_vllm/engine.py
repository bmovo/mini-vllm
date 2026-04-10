from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .model import DecoderOnlyTransformer, PastKeyValue


@dataclass(slots=True)
class GenerationResult:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    created_at: float
    first_token_at: float | None
    finished_at: float

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.created_at

    @property
    def latency_seconds(self) -> float:
        return self.finished_at - self.created_at


@dataclass(slots=True)
class _RequestState:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    created_at: float
    output_token_ids: list[int] = field(default_factory=list)
    past_key_values: tuple[PastKeyValue, ...] | None = None
    pending_input_id: int | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    @property
    def cached_tokens(self) -> int:
        if self.past_key_values is None:
            return 0
        return self.past_key_values[0][0].size(-2)

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None

    def to_result(self) -> GenerationResult:
        if self.finished_at is None:
            raise RuntimeError("request is not finished")
        return GenerationResult(
            request_id=self.request_id,
            prompt_token_ids=list(self.prompt_token_ids),
            output_token_ids=list(self.output_token_ids),
            created_at=self.created_at,
            first_token_at=self.first_token_at,
            finished_at=self.finished_at,
        )


class MiniLLMEngine:
    def __init__(
        self,
        model: DecoderOnlyTransformer,
        *,
        eos_token_id: int | None = None,
        max_batch_size: int = 8,
        device: str | torch.device | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

        self.model = model
        self.model.eval()
        self.eos_token_id = eos_token_id
        self.max_batch_size = max_batch_size
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)

        self._waiting: deque[_RequestState] = deque()
        self._active: deque[_RequestState] = deque()
        self._finished: list[GenerationResult] = []

    def submit(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        request_id: str | None = None,
    ) -> str:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        state = _RequestState(
            request_id=request_id or str(uuid.uuid4()),
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            created_at=time.perf_counter(),
        )
        self._waiting.append(state)
        return state.request_id

    def submit_many(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        return [
            self.submit(prompt, max_new_tokens=max_new_tokens)
            for prompt in prompts
        ]

    def has_pending_requests(self) -> bool:
        return bool(self._waiting or self._active)

    def step(self) -> list[GenerationResult]:
        finished_before = len(self._finished)

        decode_batch_size = min(len(self._active), self.max_batch_size)
        decode_batch = [self._active.popleft() for _ in range(decode_batch_size)]
        prefill_capacity = self.max_batch_size - decode_batch_size
        prefill_batch = [self._waiting.popleft() for _ in range(min(len(self._waiting), prefill_capacity))]

        if decode_batch:
            self._decode_batch(decode_batch)
        if prefill_batch:
            self._prefill_batch(prefill_batch)

        return self._finished[finished_before:]

    def run(self) -> list[GenerationResult]:
        while self.has_pending_requests():
            self.step()
        return list(self._finished)

    def finished_results(self) -> list[GenerationResult]:
        return list(self._finished)

    @torch.inference_mode()
    def _prefill_batch(self, states: list[_RequestState]) -> None:
        input_ids, attention_mask, prompt_lengths = self._build_prompt_batch(states)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("model returned no cache during prefill")

        per_request_cache = self._split_past_key_values(
            outputs.past_key_values,
            prompt_lengths,
            align_right=False,
        )
        logits = outputs.logits

        for batch_index, state in enumerate(states):
            state.past_key_values = per_request_cache[batch_index]
            last_prompt_index = prompt_lengths[batch_index] - 1
            next_token = int(torch.argmax(logits[batch_index, last_prompt_index]).item())
            self._record_token(state, next_token)

    @torch.inference_mode()
    def _decode_batch(self, states: list[_RequestState]) -> None:
        input_ids = torch.tensor(
            [[state.pending_input_id] for state in states],
            dtype=torch.long,
            device=self.device,
        )
        batched_past_key_values, past_lengths = self._merge_past_key_values(states)
        attention_mask = self._build_decode_attention_mask(past_lengths)
        position_ids = torch.tensor(
            past_lengths,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=batched_past_key_values,
            use_cache=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("model returned no cache during decode")

        next_past_lengths = [length + 1 for length in past_lengths]
        per_request_cache = self._split_past_key_values(
            outputs.past_key_values,
            next_past_lengths,
            align_right=True,
        )
        logits = outputs.logits[:, -1, :]

        for batch_index, state in enumerate(states):
            state.past_key_values = per_request_cache[batch_index]
            next_token = int(torch.argmax(logits[batch_index]).item())
            self._record_token(state, next_token)

    def _record_token(self, state: _RequestState, next_token: int) -> None:
        now = time.perf_counter()
        state.output_token_ids.append(next_token)
        state.pending_input_id = next_token
        if state.first_token_at is None:
            state.first_token_at = now

        should_finish = len(state.output_token_ids) >= state.max_new_tokens
        if self.eos_token_id is not None and next_token == self.eos_token_id:
            should_finish = True

        if should_finish:
            state.finished_at = now
            self._finished.append(state.to_result())
        else:
            self._active.append(state)

    def _build_prompt_batch(
        self,
        states: list[_RequestState],
    ) -> tuple[Tensor, Tensor, list[int]]:
        prompt_lengths = [len(state.prompt_token_ids) for state in states]
        max_prompt_length = max(prompt_lengths)
        batch_size = len(states)

        input_ids = torch.zeros(
            batch_size,
            max_prompt_length,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            batch_size,
            max_prompt_length,
            dtype=torch.bool,
            device=self.device,
        )

        for batch_index, state in enumerate(states):
            length = prompt_lengths[batch_index]
            input_ids[batch_index, :length] = torch.tensor(
                state.prompt_token_ids,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask[batch_index, :length] = True

        return input_ids, attention_mask, prompt_lengths

    def _build_decode_attention_mask(self, past_lengths: list[int]) -> Tensor:
        batch_size = len(past_lengths)
        total_length = max(past_lengths) + 1
        attention_mask = torch.zeros(
            batch_size,
            total_length,
            dtype=torch.bool,
            device=self.device,
        )
        for batch_index, past_length in enumerate(past_lengths):
            past_start = total_length - 1 - past_length
            attention_mask[batch_index, past_start:-1] = True
            attention_mask[batch_index, -1] = True
        return attention_mask

    def _merge_past_key_values(
        self,
        states: list[_RequestState],
    ) -> tuple[tuple[PastKeyValue, ...], list[int]]:
        if not states or states[0].past_key_values is None:
            raise RuntimeError("decode batch requires prefetched kv cache")

        num_layers = len(states[0].past_key_values)
        past_lengths = [state.cached_tokens for state in states]
        max_past_length = max(past_lengths)

        merged_layers: list[PastKeyValue] = []
        for layer_index in range(num_layers):
            layer_keys: list[Tensor] = []
            layer_values: list[Tensor] = []
            for state in states:
                if state.past_key_values is None:
                    raise RuntimeError("missing past_key_values in active request")
                key, value = state.past_key_values[layer_index]
                head_count = key.size(1)
                head_dim = key.size(-1)
                padded_key = torch.zeros(
                    1,
                    head_count,
                    max_past_length,
                    head_dim,
                    dtype=key.dtype,
                    device=key.device,
                )
                padded_value = torch.zeros(
                    1,
                    head_count,
                    max_past_length,
                    head_dim,
                    dtype=value.dtype,
                    device=value.device,
                )
                start = max_past_length - key.size(-2)
                padded_key[:, :, start:, :] = key
                padded_value[:, :, start:, :] = value
                layer_keys.append(padded_key)
                layer_values.append(padded_value)

            merged_layers.append((torch.cat(layer_keys, dim=0), torch.cat(layer_values, dim=0)))

        return tuple(merged_layers), past_lengths

    def _split_past_key_values(
        self,
        batched_past_key_values: tuple[PastKeyValue, ...],
        valid_lengths: list[int],
        *,
        align_right: bool,
    ) -> list[tuple[PastKeyValue, ...]]:
        per_request: list[list[PastKeyValue]] = [[] for _ in valid_lengths]
        for layer_key, layer_value in batched_past_key_values:
            total_length = layer_key.size(-2)
            for batch_index, valid_length in enumerate(valid_lengths):
                start = total_length - valid_length if align_right else 0
                per_request[batch_index].append(
                    (
                        layer_key[
                            batch_index : batch_index + 1,
                            :,
                            start : start + valid_length,
                            :,
                        ].contiguous(),
                        layer_value[
                            batch_index : batch_index + 1,
                            :,
                            start : start + valid_length,
                            :,
                        ].contiguous(),
                    )
                )
        return [tuple(layer_entries) for layer_entries in per_request]
