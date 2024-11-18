import torch
import torch.nn as nn


class QNetwork(nn.Module):
	def __init__(self, param, obs, device=None):
		super().__init__()

		(seq_len, Ninstr, Nsn0) = obs['sequence_numerical'].shape
		self.sn_net = 


		if device:
			self.to(device)



	def forward(self, inputs):
