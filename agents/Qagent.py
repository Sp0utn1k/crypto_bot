import copy

import torch
import torch.optim as optim

from ..utils import ReplayMemory

class Qagent:
	def __init__(self, Qnet, params):

		self.random_action = params['random_action']
		device = torch.device("cuda" if torch.cuda.is_available() and params.get('use_gpu', True) else "cpu")
		self.device = params.get('device', device)

		self.tau = params['tau']
		
		self.policy_net = Qnet(params, device=self.device)
		self.target_net = Qnet(params, device=self.device)
		self._sync_nets()

		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-3, amsgrad=True)

		self.memory = ReplayMemory(params['buffer_size'])
		self.batch_size = params['batch_size']

	def _sync_nets(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())

	def Qnet_soft_update(self):
		target_net_state_dict = self.target_net.state_dict()
		policy_net_state_dict = self.policy_net.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
		self.target_net.load_state_dict(target_net_state_dict)

	def select_action(self, state):
		return self.greedy_action(state)

	def greedy_action(self, state):
		with torch.no_grad():
			return self.policy_net(state).max(1).indices
		
	def add_to_memory(self, *args):
		self.memory.push(*args)

	def train(self):
		if len(self.memory) < self.batch_size:
			return
		transitions = self.memory.sample(self.batch_size)