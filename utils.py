from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('S', 'A', 'R', 'S_', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def greedy_f(policy_net, state):
    return policy_net(state).flatten(start_dim=1).max(1).indices#.view(1,1)

def get_random_f(action_space):
    def random_f():
        return torch.tensor([[action_space.sample()]])
    return random_f

def epsilon_greedy(state, eps, greedy_f, random_f):
    if random.random() > eps:
        with torch.no_grad():
            return greedy_f(state)
    else:
        return random_f().to(state.device)