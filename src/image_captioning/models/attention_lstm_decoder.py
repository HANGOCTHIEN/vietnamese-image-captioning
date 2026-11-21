from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..data.tokenizer_vi import VietnameseTokenizer
from .cnn_encoder import CNNEncoder


@dataclass(frozen=True)
class SpecialTokens:
    """
    Đồng bộ trực tiếp chỉ số token đặc biệt từ VietnameseTokenizer của dự án:
    pad=0 (<pad>), bos/sos=1 (<bos>), eos=2 (<eos>), unk=3 (<unk>).
    """

    PAD_TOKEN: str = VietnameseTokenizer.special_tokens["pad"]
    BOS_TOKEN: str = VietnameseTokenizer.special_tokens["bos"]
    EOS_TOKEN: str = VietnameseTokenizer.special_tokens["eos"]
    UNK_TOKEN: str = VietnameseTokenizer.special_tokens["unk"]

    PAD_IDX: int = VietnameseTokenizer.pad_id   # 0
    BOS_IDX: int = VietnameseTokenizer.bos_id   # 1
    SOS_IDX: int = VietnameseTokenizer.bos_id   # 1 (alias cho BOS_IDX)
    EOS_IDX: int = VietnameseTokenizer.eos_id   # 2
    UNK_IDX: int = VietnameseTokenizer.unk_id   # 3


@contextmanager
def _preserve_eval_mode(module: nn.Module):
    """
    Chuyển tạm thời mô hình sang eval() để tắt Dropout khi sinh caption,
    sau đó khôi phục nguyên trạng thái train/eval riêng của từng submodule
    (giữ nguyên trạng thái eval() của backbone khi freeze).
    """
    states = {sub: sub.training for sub in module.modules()}
    try:
        module.eval()
        yield
    finally:
        for sub, was_training in states.items():
            sub.training = was_training


class BahdanauAttention(nn.Module):
    """Cơ chế Soft Attention (Xu et al., 2015 - Show, Attend and Tell)."""

    def __init__(
        self,
        encoder_dim: int = CNNEncoder.encoder_dim,
        decoder_dim: int = 512,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        spatial_features: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        att1 = self.encoder_att(spatial_features)                   # [B, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)        # [B, 1, attention_dim]
        scores = self.full_att(self.tanh(att1 + att2)).squeeze(2)   # [B, num_pixels]

        alpha = self.softmax(scores)                                # [B, num_pixels]
        context = (spatial_features * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]
        return context, alpha


class AttentionLSTMDecoder(nn.Module):
    """
    Bộ giải mã LSTM tích hợp Bahdanau Attention.
    Tương thích trực tiếp với CNNEncoder, VietnameseTokenizer và KTVICDataset.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        decoder_dim: int = 512,
        attention_dim: int = 256,
        encoder_dim: int = CNNEncoder.encoder_dim,
        pad_idx: int = SpecialTokens.PAD_IDX,
        sos_idx: int = SpecialTokens.BOS_IDX,
        eos_idx: int = SpecialTokens.EOS_IDX,
        dropout: float = 0.5,
        use_attention: bool = True,
        use_deep_output: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.use_attention = use_attention
        self.use_deep_output = use_deep_output

        self.attention = (
            BahdanauAttention(
                encoder_dim=encoder_dim,
                decoder_dim=decoder_dim,
                attention_dim=attention_dim,
            )
            if use_attention
            else None
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=dropout)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        if use_deep_output:
            self.deep_output = nn.Linear(
                decoder_dim + encoder_dim + embed_dim, embed_dim
            )
            self.fc = nn.Linear(embed_dim, vocab_size)
        else:
            self.fc = nn.Linear(decoder_dim, vocab_size)

    def _validate_encoder_outputs(
        self,
        global_features: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> None:
        if global_features.ndim != 2 or global_features.shape[1] != self.encoder_dim:
            raise ValueError(
                f"global_features phải có shape [B, {self.encoder_dim}], "
                f"nhận được {tuple(global_features.shape)}."
            )
        if spatial_features.ndim != 3 or spatial_features.shape[2] != self.encoder_dim:
            raise ValueError(
                f"spatial_features phải có shape [B, num_pixels, {self.encoder_dim}], "
                f"nhận được {tuple(spatial_features.shape)}."
            )
        if global_features.shape[0] != spatial_features.shape[0]:
            raise ValueError(
                "Batch size không khớp giữa global_features và spatial_features: "
                f"{global_features.shape[0]} vs {spatial_features.shape[0]}."
            )

    def _validate_captions(
        self,
        captions: torch.Tensor,
        global_features: torch.Tensor,
    ) -> None:
        if captions.ndim != 2 or captions.size(1) < 2:
            raise ValueError(
                "captions phải có shape [B, max_len] với max_len >= 2, "
                f"nhận được {tuple(captions.shape)}."
            )
        if captions.size(0) != global_features.size(0):
            raise ValueError(
                "Batch size của captions và encoder outputs không khớp: "
                f"{captions.size(0)} vs {global_features.size(0)}."
            )
        if captions.dtype != torch.long:
            raise TypeError(
                f"captions phải có dtype torch.long, nhận được {captions.dtype}."
            )
        if captions.device != global_features.device:
            raise ValueError(
                "captions và encoder outputs phải nằm trên cùng device: "
                f"{captions.device} vs {global_features.device}."
            )
        if (captions < 0).any() or (captions >= self.vocab_size).any():
            raise ValueError(
                f"captions chứa token ID ngoài phạm vi hợp lệ [0, {self.vocab_size - 1}]."
            )
        if not torch.all(captions[:, 0] == self.sos_idx):
            raise ValueError(
                f"Mỗi caption phải bắt đầu bằng BOS/SOS (ID={self.sos_idx})."
            )
        if not torch.all((captions == self.eos_idx).any(dim=1)):
            raise ValueError(
                f"Mỗi caption phải chứa ít nhất một token EOS (ID={self.eos_idx})."
            )

    def init_hidden_state(
        self, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.init_h(global_features))
        c = torch.tanh(self.init_c(global_features))
        return h, c

    def forward_step(
        self,
        prev_words: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor],
        spatial_features: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        h, c = hidden_state
        embeddings = self.dropout(self.embedding(prev_words))

        if self.use_attention and self.attention is not None:
            context, alpha = self.attention(spatial_features, h)
        else:
            context = spatial_features.mean(dim=1)
            alpha = spatial_features.new_zeros(
                spatial_features.size(0), spatial_features.size(1)
            )

        lstm_input = torch.cat([embeddings, context], dim=1)
        h_next, c_next = self.decode_step(lstm_input, (h, c))

        if self.use_deep_output:
            combined = torch.cat([h_next, context, embeddings], dim=1)
            out_features = torch.tanh(self.deep_output(self.dropout(combined)))
            logits = self.fc(self.dropout(out_features))
        else:
            logits = self.fc(self.dropout(h_next))

        return logits, (h_next, c_next), alpha

    def forward(
        self,
        global_features: torch.Tensor,
        spatial_features: torch.Tensor,
        captions: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            captions: Truyền trực tiếp batch['caption_ids'] từ collate_fn, shape [B, max_len].
        """
        self._validate_encoder_outputs(global_features, spatial_features)
        self._validate_captions(captions, global_features)
        if not (0.0 <= teacher_forcing_ratio <= 1.0):
            raise ValueError(
                f"teacher_forcing_ratio phải nằm trong [0.0, 1.0], nhận được {teacher_forcing_ratio}."
            )

        batch_size = global_features.size(0)
        num_pixels = spatial_features.size(1)
        decode_len = captions.size(1) - 1

        h, c = self.init_hidden_state(global_features)

        predictions = global_features.new_zeros(
            batch_size, decode_len, self.vocab_size
        )
        alphas = spatial_features.new_zeros(batch_size, decode_len, num_pixels)

        input_word = captions[:, 0]

        for t in range(decode_len):
            logits, (h, c), alpha = self.forward_step(
                input_word, (h, c), spatial_features
            )
            predictions[:, t, :] = logits
            alphas[:, t, :] = alpha

            use_tf = (
                torch.rand(batch_size, device=global_features.device)
                < teacher_forcing_ratio
            )
            predicted_words = logits.argmax(dim=-1)
            input_word = torch.where(use_tf, captions[:, t + 1], predicted_words)

        return predictions, alphas

    def compute_doubly_stochastic_loss(
        self,
        alphas: torch.Tensor,
        captions: torch.Tensor,
        pad_idx: int | None = None,
        lambda_c: float = 1.0,
    ) -> torch.Tensor:
        """
        Tính phạt Doubly Stochastic Attention: lambda_c * sum_i (1 - sum_t alpha_{t,i})^2.
        Tự động loại bỏ các bước <pad> khỏi tổng theo trục thời gian t.
        """
        if lambda_c < 0:
            raise ValueError(f"lambda_c phải >= 0, nhận được {lambda_c}.")
        if alphas.ndim != 3:
            raise ValueError(
                f"alphas phải có shape [B, L-1, N], nhận được {tuple(alphas.shape)}."
            )
        if captions.ndim != 2:
            raise ValueError(
                f"captions phải có shape [B, L], nhận được {tuple(captions.shape)}."
            )
        if alphas.shape[:2] != (captions.size(0), captions.size(1) - 1):
            raise ValueError(
                f"Shape của alphas {tuple(alphas.shape[:2])} và captions "
                f"{(captions.size(0), captions.size(1) - 1)} không khớp."
            )

        if not self.use_attention:
            return alphas.new_zeros(())

        effective_pad = self.pad_idx if pad_idx is None else pad_idx
        valid_mask = (captions[:, 1:] != effective_pad).to(alphas.dtype).unsqueeze(-1)
        masked_alphas = alphas * valid_mask
        region_attention_sum = masked_alphas.sum(dim=1)
        penalty = ((1.0 - region_attention_sum) ** 2).sum(dim=1).mean()
        return lambda_c * penalty

    def trim_sequences(self, token_ids: torch.Tensor) -> list[list[int]]:
        """
        Cắt sạch từng chuỗi tại <eos> đầu tiên và loại bỏ <bos>, <pad>
        trước khi gọi vocab.decode() của VietnameseTokenizer.
        """
        cleaned_batch: list[list[int]] = []
        for seq in token_ids.tolist():
            cleaned: list[int] = []
            for idx in seq:
                if idx == self.eos_idx:
                    break
                if idx in (self.sos_idx, self.pad_idx):
                    continue
                cleaned.append(int(idx))
            cleaned_batch.append(cleaned)
        return cleaned_batch

    def decode_predictions(
        self,
        token_ids: torch.Tensor,
        vocab: VietnameseTokenizer,
    ) -> list[str]:
        """Cắt <eos> theo từng mẫu rồi chuyển sang chuỗi tiếng Việt qua vocab.decode()."""
        trimmed = self.trim_sequences(token_ids)
        return [vocab.decode(seq, skip_special_tokens=True) for seq in trimmed]

    @torch.inference_mode()
    def generate_greedy(
        self,
        global_features: torch.Tensor,
        spatial_features: torch.Tensor,
        max_len: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_len < 1:
            raise ValueError(f"max_len phải >= 1, nhận được {max_len}.")
        self._validate_encoder_outputs(global_features, spatial_features)

        with _preserve_eval_mode(self):
            batch_size = global_features.size(0)
            device = global_features.device

            h, c = self.init_hidden_state(global_features)
            input_word = torch.full(
                (batch_size,), self.sos_idx, dtype=torch.long, device=device
            )

            sampled_ids = []
            alphas = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                logits, (h, c), alpha = self.forward_step(
                    input_word, (h, c), spatial_features
                )
                predicted = logits.argmax(dim=-1)

                predicted = torch.where(
                    finished,
                    torch.full_like(predicted, self.pad_idx),
                    predicted,
                )

                sampled_ids.append(predicted)
                alphas.append(alpha)

                finished |= predicted == self.eos_idx
                if finished.all():
                    break

                input_word = predicted

            return torch.stack(sampled_ids, dim=1), torch.stack(alphas, dim=1)

    @torch.inference_mode()
    def generate_beam_search(
        self,
        global_features: torch.Tensor,
        spatial_features: torch.Tensor,
        beam_size: int = 3,
        max_len: int = 30,
        length_penalty: float = 0.7,
    ) -> tuple[list[int], torch.Tensor]:
        if beam_size < 1 or beam_size > self.vocab_size:
            raise ValueError(
                f"beam_size phải nằm trong [1, {self.vocab_size}], nhận được {beam_size}."
            )
        if max_len < 1:
            raise ValueError(f"max_len phải >= 1, nhận được {max_len}.")
        if length_penalty < 0:
            raise ValueError(
                f"length_penalty phải >= 0, nhận được {length_penalty}."
            )

        self._validate_encoder_outputs(global_features, spatial_features)
        if global_features.size(0) != 1:
            raise ValueError(
                "generate_beam_search chỉ hỗ trợ từng ảnh một (batch_size=1), "
                f"nhận được batch_size={global_features.size(0)}."
            )

        with _preserve_eval_mode(self):
            k = beam_size
            num_pixels = spatial_features.size(1)
            device = global_features.device

            encoder_out = spatial_features.expand(k, num_pixels, self.encoder_dim)
            h, c = self.init_hidden_state(global_features.expand(k, self.encoder_dim))

            k_prev_words = torch.full(
                (k,), self.sos_idx, dtype=torch.long, device=device
            )
            seqs = k_prev_words.unsqueeze(1)
            seqs_alpha = spatial_features.new_zeros(k, 0, num_pixels)
            top_k_scores = global_features.new_zeros(k, 1)

            complete_seqs: list[list[int]] = []
            complete_seqs_alpha: list[torch.Tensor] = []
            complete_seqs_scores: list[float] = []

            for step in range(max_len):
                logits, (h, c), alpha = self.forward_step(
                    k_prev_words, (h, c), encoder_out
                )
                log_probs = F.log_softmax(logits, dim=-1)
                scores = top_k_scores.expand_as(log_probs) + log_probs

                if step == 0:
                    top_k_scores, top_k_words = scores[0].topk(k, dim=0)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)

                prev_beam_inds = top_k_words // self.vocab_size
                next_word_inds = top_k_words % self.vocab_size

                seqs = torch.cat(
                    [seqs[prev_beam_inds], next_word_inds.unsqueeze(1)], dim=1
                )
                seqs_alpha = torch.cat(
                    [seqs_alpha[prev_beam_inds], alpha[prev_beam_inds].unsqueeze(1)],
                    dim=1,
                )

                incomplete_inds = [
                    ind
                    for ind, next_word in enumerate(next_word_inds)
                    if next_word.item() != self.eos_idx
                ]
                complete_inds = list(
                    set(range(len(next_word_inds))) - set(incomplete_inds)
                )

                if complete_inds:
                    for ind in complete_inds:
                        seq_list = seqs[ind].tolist()
                        norm_score = top_k_scores[ind].item() / (
                            max(1, len(seq_list) - 1) ** length_penalty
                        )
                        complete_seqs.append(seq_list)
                        complete_seqs_alpha.append(seqs_alpha[ind])
                        complete_seqs_scores.append(norm_score)

                k -= len(complete_inds)
                if k == 0:
                    break

                seqs = seqs[incomplete_inds]
                seqs_alpha = seqs_alpha[incomplete_inds]
                h = h[prev_beam_inds[incomplete_inds]]
                c = c[prev_beam_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_beam_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds]

            if k > 0:
                for ind in range(seqs.size(0)):
                    seq_list = seqs[ind].tolist()
                    norm_score = top_k_scores[ind].item() / (
                        max(1, len(seq_list) - 1) ** length_penalty
                    )
                    complete_seqs.append(seq_list)
                    complete_seqs_alpha.append(seqs_alpha[ind])
                    complete_seqs_scores.append(norm_score)

            best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            raw_best_seq = complete_seqs[best_idx]
            raw_best_alpha = complete_seqs_alpha[best_idx]

            clean_seq: list[int] = []
            valid_steps = 0
            for step_idx, tok in enumerate(raw_best_seq[1:]):
                if tok == self.eos_idx:
                    break
                clean_seq.append(int(tok))
                valid_steps = step_idx + 1

            return clean_seq, raw_best_alpha[:valid_steps]


class AttentionLSTMCaptioner(nn.Module):
    """Mô hình End-to-End ghép CNNEncoder và AttentionLSTMDecoder."""

    def __init__(
        self,
        encoder: CNNEncoder,
        decoder: AttentionLSTMDecoder,
    ) -> None:
        super().__init__()
        if encoder.encoder_dim != decoder.encoder_dim:
            raise ValueError(
                f"encoder_dim không khớp giữa Encoder ({encoder.encoder_dim}) "
                f"và Decoder ({decoder.encoder_dim})."
            )
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_features, spatial_features = self.encoder(images)
        return self.decoder(
            global_features=global_features,
            spatial_features=spatial_features,
            captions=captions,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

    @torch.inference_mode()
    def generate_greedy(
        self,
        images: torch.Tensor,
        max_len: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _preserve_eval_mode(self):
            global_features, spatial_features = self.encoder(images)
            return self.decoder.generate_greedy(
                global_features=global_features,
                spatial_features=spatial_features,
                max_len=max_len,
            )

    @torch.inference_mode()
    def generate_beam_search(
        self,
        image: torch.Tensor,
        beam_size: int = 3,
        max_len: int = 30,
        length_penalty: float = 0.7,
    ) -> tuple[list[int], torch.Tensor]:
        with _preserve_eval_mode(self):
            global_features, spatial_features = self.encoder(image)
            return self.decoder.generate_beam_search(
                global_features=global_features,
                spatial_features=spatial_features,
                beam_size=beam_size,
                max_len=max_len,
                length_penalty=length_penalty,
            )