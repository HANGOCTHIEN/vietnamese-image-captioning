"""LSTM decoder E0, khởi tạo hidden/cell state từ global features của ảnh."""

from __future__ import annotations

import torch
from torch import nn


class LSTMDecoder(nn.Module):
    """Sinh raw next-token logits bằng LSTM một chiều có conditioning từ ảnh.

    Hai Linear độc lập và tanh đưa global features [B, encoder_dim] thành
    h0/c0 [num_layers, B, hidden_dim]. Token IDs [B, T] đi qua Embedding,
    embedding dropout, LSTM, output dropout và vocabulary projection để tạo
    logits [B, T, vocab_size]. Không có attention hoặc xử lý văn bản trong
    decoder; vocab_size và pad_token_id do bên gọi cung cấp.

    Forward dùng toàn bộ input_ids đã được dịch chuỗi bởi collate_fn, không
    tự thêm BOS/EOS. Logits tại padding vẫn được giữ; loss bên ngoài cần bỏ
    qua target PAD. step() nhận/trả state tường minh để sinh một token, không
    chọn token hoặc lưu state vào model. Dropout tự tắt khi gọi eval().

    Device, dtype và random seed được quản lý từ bên ngoài. Model không tự
    di chuyển/ép kiểu input và luôn giữ gradient qua image conditioning.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 2048,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("vocab_size", vocab_size), ("encoder_dim", encoder_dim),
            ("embedding_dim", embedding_dim), ("hidden_dim", hidden_dim),
            ("num_layers", num_layers),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} phải là số nguyên, không nhận bool.")
            if value < 1:
                raise ValueError(f"{name} phải >= 1, nhận được {value}.")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout phải là số thực trong [0, 1).")
        if not 0 <= dropout < 1:
            raise ValueError("dropout phải thỏa mãn 0 <= dropout < 1.")
        if type(pad_token_id) is not int:
            raise TypeError("pad_token_id phải là số nguyên, không nhận bool.")
        if not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id phải thỏa mãn 0 <= pad_token_id < vocab_size.")

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.image_to_hidden = nn.Linear(encoder_dim, num_layers * hidden_dim)
        self.image_to_cell = nn.Linear(encoder_dim, num_layers * hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def _validate_features(self, global_features: torch.Tensor) -> None:
        """Kiểm tra global vector, không tự gộp hoặc flatten spatial features."""
        if not isinstance(global_features, torch.Tensor):
            raise TypeError("global_features phải là torch.Tensor.")
        if global_features.ndim != 2 or global_features.shape[1] != self.encoder_dim:
            raise ValueError(
                f"global_features phải có shape [B, {self.encoder_dim}], "
                f"nhận được {tuple(global_features.shape)}; không nhận spatial features."
            )
        if global_features.shape[0] < 1:
            raise ValueError("global_features phải có batch size B >= 1.")
        if not global_features.is_floating_point():
            raise TypeError("global_features phải là floating-point tensor.")
        parameter = self.image_to_hidden.weight
        if global_features.dtype != parameter.dtype:
            raise TypeError(
                f"global_features có dtype {global_features.dtype}, "
                f"nhưng decoder yêu cầu {parameter.dtype}."
            )
        if global_features.device != parameter.device:
            raise ValueError("global_features và decoder phải ở cùng device.")

    def _validate_tokens(self, token_ids: torch.Tensor, ndim: int, name: str) -> None:
        """Kiểm tra shape, int64, device và khoảng ID trước khi lookup embedding."""
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError(f"{name} phải là torch.Tensor.")
        shape = "[B, T]" if ndim == 2 else "[B]"
        if token_ids.ndim != ndim:
            raise ValueError(f"{name} phải có shape {shape}, nhận được {tuple(token_ids.shape)}.")
        if any(size < 1 for size in token_ids.shape):
            raise ValueError(f"{name} không được có batch hoặc sequence length rỗng.")
        if token_ids.dtype != torch.long:
            raise TypeError(f"{name} phải có dtype torch.long (int64), nhận được {token_ids.dtype}.")
        if token_ids.device != self.embedding.weight.device:
            raise ValueError(f"{name} và decoder phải ở cùng device.")
        if ((token_ids < 0) | (token_ids >= self.vocab_size)).any():
            raise ValueError(f"{name} phải chứa ID trong khoảng 0 <= token_id < {self.vocab_size}.")

    def _validate_state(
        self, state: tuple[torch.Tensor, torch.Tensor], batch_size: int
    ) -> None:
        """Kiểm tra cả h/c theo số tầng, batch, hidden_dim, dtype và device."""
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("state phải là tuple (h, c) gồm hai tensor.")
        expected = (self.num_layers, batch_size, self.hidden_dim)
        parameter = self.lstm.weight_ih_l0
        for name, tensor in zip(("h", "c"), state):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"state.{name} phải là torch.Tensor.")
            if tuple(tensor.shape) != expected:
                raise ValueError(f"state.{name} phải có shape {expected}, nhận được {tuple(tensor.shape)}.")
            if not tensor.is_floating_point() or tensor.dtype != parameter.dtype:
                raise TypeError(f"state.{name} phải là floating-point tensor có dtype {parameter.dtype}.")
            if tensor.device != parameter.device:
                raise ValueError(f"state.{name} và decoder phải ở cùng device.")

    def init_state(
        self, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tạo (h0, c0) từ ảnh, mỗi tensor có shape [num_layers, B, hidden_dim].

        Input phải là global features số thực [B, encoder_dim], có cùng dtype
        và device với decoder. Hai projection học độc lập; reshape/transpose
        bảo toàn batch size 1 và gradient về features cũng như các parameters.
        """
        self._validate_features(global_features)
        batch_size = global_features.shape[0]
        hidden = torch.tanh(self.image_to_hidden(global_features))
        cell = torch.tanh(self.image_to_cell(global_features))
        hidden = hidden.reshape(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        cell = cell.reshape(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return hidden, cell

    def _decode(
        self, input_ids: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Dùng chung đường embedding/LSTM/projection cho cả forward và step."""
        embeddings = self.embedding_dropout(self.embedding(input_ids))
        outputs, new_state = self.lstm(embeddings, state)
        logits = self.output_projection(self.output_dropout(outputs))
        return logits, new_state

    def forward(
        self, global_features: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Teacher forcing: [B, encoder_dim] và int64 [B, T] → logits [B, T, V].

        input_ids đã chứa BOS/shift phù hợp từ collate_fn. Không thêm special
        token, không tạo targets, không sampling, softmax hoặc tính loss.
        LSTM chạy cả chuỗi trong một lần gọi, kể cả các vị trí padding.
        """
        self._validate_features(global_features)
        self._validate_tokens(input_ids, 2, "input_ids")
        if global_features.shape[0] != input_ids.shape[0]:
            raise ValueError("Batch size của global_features và input_ids không khớp.")
        logits, _ = self._decode(input_ids, self.init_state(global_features))
        return logits

    def step(
        self, token_ids: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Xử lý một token int64 [B], trả logits [B, V] và state (h, c) mới.

        State vào/ra có shape [num_layers, B, hidden_dim]. Bên gọi truyền tiếp
        state và tự chọn token/EOS. Dùng eval() cùng no_grad() từ bên ngoài
        khi inference; phương thức này vẫn hỗ trợ gradient nếu bên gọi cần.
        """
        self._validate_tokens(token_ids, 1, "token_ids")
        self._validate_state(state, token_ids.shape[0])
        logits, new_state = self._decode(token_ids.unsqueeze(1), state)
        return logits[:, 0, :], new_state
