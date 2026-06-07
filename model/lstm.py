from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import CNNEncoder


class RNNDecoder(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		embed_size: int,
		num_layers: int,
		dropout: float,
	):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.num_layers = num_layers
		self.hidden_size = embed_size
		self.rnn = nn.LSTM(
			input_size=embed_size,
			hidden_size=embed_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.init_hidden = nn.Linear(embed_size, embed_size)
		self.init_cell = nn.Linear(embed_size, embed_size)
		self.fc_out = nn.Linear(embed_size, vocab_size)

		self._init_weights()

	def _init_weights(self) -> None:
		nn.init.xavier_uniform_(self.embed.weight)
		nn.init.xavier_uniform_(self.fc_out.weight)
		nn.init.zeros_(self.fc_out.bias)
		nn.init.xavier_uniform_(self.init_hidden.weight)
		nn.init.zeros_(self.init_hidden.bias)
		nn.init.xavier_uniform_(self.init_cell.weight)
		nn.init.zeros_(self.init_cell.bias)

	def _init_state(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		context = memory.mean(dim=1)
		hidden = torch.tanh(self.init_hidden(context))
		cell = torch.tanh(self.init_cell(context))
		hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
		cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
		return hidden, cell

	def forward(
		self,
		tgt: torch.Tensor,
		memory: torch.Tensor,
		tgt_key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		del tgt_key_padding_mask
		tgt_emb = self.embed(tgt)
		hidden = self._init_state(memory)
		out, _ = self.rnn(tgt_emb, hidden)
		return self.fc_out(out)


class ImageCaptioningModel(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		embed_size: int = 512,
		num_heads: int = 8,
		num_layers: int = 3,
		ff_dim: int = 2048,
		dropout: float = 0.1,
		max_len: int = 512,
		fine_tune_cnn: bool = True,
	):
		super().__init__()
		del num_heads, ff_dim, max_len
		self.encoder = CNNEncoder(embed_size, fine_tune=fine_tune_cnn)
		self.decoder = RNNDecoder(
			vocab_size=vocab_size,
			embed_size=embed_size,
			num_layers=num_layers,
			dropout=dropout,
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
		logits = self.decoder(tgt_in, memory, tgt_key_padding_mask=pad_mask)
		return logits

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

			candidates.sort(key=lambda x: x[0], reverse=True)
			beams = []
			for s, toks in candidates[:beam_size]:
				if toks[-1] == vocab.EOS:
					done.append((s / len(toks), toks))
				else:
					beams.append((s, toks))
			if not beams:
				break

		if done:
			done.sort(key=lambda x: x[0], reverse=True)
			best_tokens = done[0][1]
		else:
			best_tokens = max(beams, key=lambda x: x[0])[1]

		return vocab.decode(best_tokens[1:]).split()
