import torch
from market_env import MarketEnv

from .models.transformer_model_v1 import QNetwork
from .agents.Qagent import Qagent

env = MarketEnv(external_config_dir='crypto_bot/configs', config_name='default')
obs, _ = env.reset()

print([k for k in obs.keys()])


# params = {}
# params['N_obs'] = N_obs
# params['N_instr'] = N_instr
# params['embedding_dim'] = N_instr * 25 # Check division by N_instr always integer
# params['seq_len'] = env.window_size
# params['Nout'] = 9
# params['random_action'] = env.action_space.sample
# params['tau'] = 0.01
# params['buffer_size'] = 10000
# params['batch_size'] = 128

# agent = Qagent(QNetwork, params)

# done = False
# S = torch.tensor(obs, device=agent.device).unsqueeze(0)
# step = 0
# while not done:
# 	step += 1
# 	print(f'\nStep {step}')
# 	print(S.shape)
# 	A = agent.select_action(S)
# 	print(A)
# 	S_, R, done, _ = env.step(A.item())
# 	R = torch.tensor(R, device=agent.device)
# 	if not done:
# 		S_ = torch.tensor(S_, device=agent.device).unsqueeze(0)

# 	agent.add_to_memory(S, A, R, S_, done)
# 	if not done:
# 		S = S_

# 	agent.Qnet_soft_update()