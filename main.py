from market_env import MarketEnv

env = MarketEnv(config_name='default')
observation, info = env.reset()
