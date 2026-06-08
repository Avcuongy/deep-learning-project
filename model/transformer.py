from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import CNNEncoder


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_size: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        self.embed_size = embed_size

        self.embed = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0,
        )

        self.pos_enc = PositionalEncoding(
            embed_size=embed_size,
            max_len=max_len,
            dropout=dropout,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_size),
        )

        self.pre_head_norm = nn.LayerNorm(embed_size)

        self.fc_out = nn.Linear(
            embed_size,
            vocab_size,
            bias=False,
        )

        self._init_weights()
        self.fc_out.weight = self.embed.weight
        self.register_buffer(
            "_causal_mask",
            torch.empty(0),
            persistent=False,
        )

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        if self.embed.padding_idx is not None:
            with torch.no_grad():
                self.embed.weight[self.embed.padding_idx].fill_(0)

    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask.numel() == 0 or self._causal_mask.size(0) < size:
            mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
            self._causal_mask = mask

        return self._causal_mask[:size, :size].to(device)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        tgt_emb = self.embed(tgt)
        tgt_emb = tgt_emb * math.sqrt(self.embed_size)
        tgt_emb = self.pos_enc(tgt_emb)

        causal_mask = self._get_causal_mask(
            tgt.size(1),
            tgt.device,
        )

        hidden = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        hidden = self.pre_head_norm(hidden)

        if return_hidden:
            return hidden

        return self.fc_out(hidden)


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        fine_tune_cnn: bool = True,
    ):
        super().__init__()
        self.encoder = CNNEncoder(embed_size, fine_tune=fine_tune_cnn)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.image_proj = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, 256),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, 256),
        )

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        pad_idx: int = 0,
    ) -> torch.Tensor:
        memory = self.encoder(images)
        tgt_in = captions[:, :-1]
        pad_mask = tgt_in == pad_idx
        return self.decoder(
            tgt=tgt_in,
            memory=memory,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=None,
        )

    def forward_with_features(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        pad_idx: int = 0,
    ):
        memory = self.encoder(images)
        tgt_in = captions[:, :-1]
        pad_mask = tgt_in == pad_idx
        hidden = self.decoder(
            tgt=tgt_in,
            memory=memory,
            tgt_key_padding_mask=pad_mask,
            return_hidden=True,
        )
        logits = self.decoder.fc_out(hidden)
        return logits, memory, hidden

    @torch.no_grad()
    def generate_caption(
        self,
        image: torch.Tensor,
        vocab,
        max_len: int = 100,
    ) -> list[str]:
        self.eval()
        memory = self.encoder(image)
        tokens = [vocab.SOS]

        for _ in range(max_len):
            tgt = torch.tensor([tokens], device=image.device)
            logits = self.decoder(tgt, memory)
            nxt = logits[0, -1].argmax().item()
            if nxt == vocab.EOS:
                break
            tokens.append(nxt)

        return vocab.decode(tokens[1:]).split()

    @torch.no_grad()
    def beam_search(
        self,
        image: torch.Tensor,
        vocab,
        beam_size: int = 5,
        max_len: int = 100,
    ) -> list[str]:
        self.eval()
        memory = self.encoder(image)
        beams = [(0.0, [vocab.SOS])]
        done = []

        for _ in range(max_len):
            candidates = []

            for score, tokens in beams:
                tgt = torch.tensor([tokens], device=image.device)
                logits = self.decoder(tgt, memory)
                log_prob = F.log_softmax(logits[0, -1], dim=-1)
                topk_lp, topk_id = log_prob.topk(beam_size)

                for lp, tid in zip(topk_lp.tolist(), topk_id.tolist()):
                    candidates.append((score + lp, tokens + [tid]))

            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = []

            for score, tokens in candidates[:beam_size]:
                if tokens[-1] == vocab.EOS:
                    done.append((score / len(tokens), tokens))
                else:
                    beams.append((score, tokens))

            if not beams:
                break

        if done:
            done.sort(key=lambda item: item[0], reverse=True)
            best_tokens = done[0][1]
        else:
            best_tokens = max(beams, key=lambda item: item[0])[1]

        return vocab.decode(best_tokens[1:]).split()
