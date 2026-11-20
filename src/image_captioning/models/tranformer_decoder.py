"""Transformer decoder cho bài toán sinh caption tiếng Việt (PyTorch).

Quy ước:
  - img_feats: (B, N, enc_dim) hoặc (B, C, H, W) từ CNN encoder (chưa pool)
  - captions : (B, T) gồm <bos> w1 w2 ... <eos> <pad>...
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerCaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        enc_dim=2048,        # số kênh đầu ra của CNN encoder
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_ff=2048,
        dropout=0.1,
        max_len=64,          # độ dài caption tối đa (kể cả bos/eos)
        max_img_tokens=256,  # số vùng ảnh tối đa (H*W của feature map)
        pad_id=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # Chiếu feature ảnh sang d_model + positional embedding học được cho vùng ảnh
        self.img_proj = nn.Sequential(
            nn.Linear(enc_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )
        self.img_pos = nn.Parameter(torch.zeros(1, max_img_tokens, d_model))
        nn.init.trunc_normal_(self.img_pos, std=0.02)

        # Phía text
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(d_model))

        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_out.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[self.pad_id].zero_()

    # ---------- các khối con ----------
    def encode_image(self, img_feats):
        if img_feats.dim() == 4:  # (B, C, H, W) -> (B, H*W, C)
            img_feats = img_feats.flatten(2).transpose(1, 2)
        mem = self.img_proj(img_feats)
        return mem + self.img_pos[:, : mem.size(1)]

    def decode(self, tokens, memory):
        T = tokens.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        pad_mask = tokens == self.pad_id
        x = self.embed(tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        h = self.decoder(
            x, memory, tgt_mask=causal, tgt_key_padding_mask=pad_mask
        )
        return self.fc_out(h)  # (B, T, V)

    # ---------- huấn luyện ----------
    def forward(self, img_feats, captions):
        """Teacher forcing. Trả về logits (B, T-1, V) ứng với captions[:, 1:]."""
        memory = self.encode_image(img_feats)
        return self.decode(captions[:, :-1], memory)

    # ---------- suy luận ----------
    @torch.no_grad()
    def greedy_generate(self, img_feats, bos_id, eos_id, max_len=30):
        self.eval()
        memory = self.encode_image(img_feats)
        B = memory.size(0)
        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(max_len):
            next_tok = self.decode(seqs, memory)[:, -1].argmax(-1)
            next_tok = next_tok.masked_fill(done, self.pad_id)
            seqs = torch.cat([seqs, next_tok.unsqueeze(1)], 1)
            done |= next_tok == eos_id
            if done.all():
                break
        return [self._strip(s.tolist(), bos_id, eos_id) for s in seqs]

    @torch.no_grad()
    def beam_search(self, img_feats, bos_id, eos_id, beam_size=3, max_len=30, alpha=0.7):
        """Beam search cho 1 ảnh: img_feats có batch = 1. Trả về list id token."""
        self.eval()
        memory = self.encode_image(img_feats)  # (1, N, d)
        dev = memory.device
        seqs = torch.full((1, 1), bos_id, dtype=torch.long, device=dev)
        scores = torch.zeros(1, device=dev)
        finished = []
        for _ in range(max_len):
            k = seqs.size(0)
            logp = F.log_softmax(self.decode(seqs, memory.expand(k, -1, -1))[:, -1], -1)
            V = logp.size(-1)
            cand = (scores.unsqueeze(1) + logp).view(-1)
            top, idx = cand.topk(min(beam_size - len(finished), cand.numel()))
            bi, wi = idx // V, idx % V
            seqs = torch.cat([seqs[bi], wi.unsqueeze(1)], 1)
            scores = top
            end = wi == eos_id
            for s, sc in zip(seqs[end], scores[end]):
                finished.append((s.tolist(), sc.item() / (len(s) ** alpha)))
            seqs, scores = seqs[~end], scores[~end]
            if len(finished) >= beam_size or seqs.size(0) == 0:
                break
        if not finished:  # chưa gặp eos trong max_len
            finished = [(s.tolist(), sc.item() / (len(s) ** alpha)) for s, sc in zip(seqs, scores)]
        best = max(finished, key=lambda x: x[1])[0]
        return self._strip(best, bos_id, eos_id)

    @staticmethod
    def _strip(ids, bos_id, eos_id):
        out = []
        for t in ids:
            if t == bos_id:
                continue
            if t == eos_id:
                break
            out.append(t)
        return out


# ---------- Ví dụ vòng huấn luyện ----------
def train_one_epoch(encoder, decoder, loader, optimizer, device, pad_id, scheduler=None):
    encoder.train(); decoder.train()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    total = 0.0
    for images, captions in loader:
        images, captions = images.to(device), captions.to(device)
        feats = encoder(images)                      # (B, C, H, W) hoặc (B, N, C)
        logits = decoder(feats, captions)            # (B, T-1, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total += loss.item()
    return total / len(loader)


if __name__ == "__main__":
    V, B, T = 5000, 4, 20
    dec = TransformerCaptionDecoder(vocab_size=V, enc_dim=2048)
    feats = torch.randn(B, 2048, 7, 7)               # giả lập đầu ra ResNet
    caps = torch.randint(4, V, (B, T)); caps[:, 0] = 1; caps[:, -1] = 2
    print(dec(feats, caps).shape)                    # (4, 19, 5000)
    print(dec.greedy_generate(feats, bos_id=1, eos_id=2, max_len=10)[0])
    print(dec.beam_search(feats[:1], bos_id=1, eos_id=2, beam_size=3, max_len=10))