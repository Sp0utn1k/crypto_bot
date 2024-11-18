import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_len, 0, -1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class QNetwork(nn.Module):
	def __init__(self, params, device=None):
		super().__init__()

		N_obs = params['N_obs']
		N_instr = params['N_instr']
		N_embed = params['embedding_dim']
		seq_len = params['seq_len']

		N1 = N_embed // N_instr
		N2 = N1 * N_instr
		Nout = params['Nout']

		self.batchnorm = nn.BatchNorm2d(seq_len)
		self.ffn1 = nn.Linear(N_obs, N1)
		self.relu = nn.ReLU()
		self.ffn2 = nn.Linear(N2, N_embed)


		self.ffno = nn.Linear(seq_len*N_embed, Nout)

		if device:
			self.to(device)

	def forward(self,inputs):
		(B, S, N_instr, N_obs) = inputs.shape

		x = self.batchnorm(inputs)

		x = self.ffn1(x) # (B, S, N_instr, N1)
		x = self.relu(x)

		x = x.flatten(start_dim=2) # (B, S, N2)
		x = self.ffn2(x) # (B, S, N_embed)
		x = self.relu(x)

		x = x.flatten(start_dim=1) # (B, S*N_embed)
		x = self.ffno(x)
		return x
