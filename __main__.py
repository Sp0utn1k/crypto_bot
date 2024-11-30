import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from market_env import MarketEnv
from market_env import data_loader

EPS = 1e-4

def log_predictions_to_tensorboard(writer, unscaled_sequences, targets, unscaled_predictions, i, j, global_step, show_N_last=32):
    """
    Log a grid of subplots visualizing the closing rates and corresponding predictions to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        input_sequences (torch.Tensor): OHLC sequences of shape (B, W, 4).
        targets (torch.Tensor): Target values of shape (B, 1).
        predictions (torch.Tensor): Predicted values of shape (B, 1).
        max_vals (torch.Tensor): Maximum values used for rescaling of shape (B, 1).
        i (int): Number of rows in the grid.
        j (int): Number of columns in the grid.
        global_step (int): The global step value for TensorBoard logging.
    """
    batch_size, W, _ = unscaled_sequences.shape
    num_plots = min(i * j, batch_size)

    fig, axes = plt.subplots(i, j, figsize=(15, 10))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Detach tensors to numpy arrays for plotting
    sequences = unscaled_sequences.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    predictions = unscaled_predictions.detach().cpu().numpy()

    for idx in range(num_plots):
        ax = axes[idx]

        # Get closing rates from the sequences
        closing_rates = sequences[idx, -show_N_last:, -1]  # Shape: (W,)
        closing_rates = np.append(closing_rates, targets[idx, 0])  # Shape: (W+1,)

        # Get the corresponding prediction
        prediction = predictions[idx, 0]

        # Plot closing rates and predictions
        ax.plot(range(show_N_last+1), closing_rates, label='Closing Rates', marker='o')
        ax.scatter(show_N_last, prediction, color='red', label='Prediction', zorder=5)

        # Formatting
        ax.set_title(f"Sample {idx+1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Closing Rate")
        ax.legend()

    # Remove unused subplots
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    # Log the figure to TensorBoard
    writer.add_figure("Predictions", fig, global_step=global_step)

    # Close the figure to avoid memory issues
    plt.close(fig)


class TemporalDataset(Dataset):
	def __init__(self, data, window_size):
		"""
		Args:
			data (torch.Tensor): Temporal dataset of shape (T, 4).
			window_size (int): Length of each sequence (W).
		"""
		self.data = data
		self.window_size = window_size
		self.num_samples = len(data) - window_size

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		"""
		Returns:
			torch.Tensor: A window of shape (W, 4).
		"""
		input_sequence = self.data[idx:idx + self.window_size]  # Shape: (W, 4)
		target = self.data[idx + self.window_size, -1].unsqueeze(-1)  # The 4th dimension at time step t
		return input_sequence, target

class SequenceScaler(nn.Module):
	def __init__(self):
		super(SequenceScaler, self).__init__()

	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Mini-batch of shape (B, W, 4).

		Returns:
			scaled_sequences (torch.Tensor): Re-scaled sequences of shape (B, W, 4).
			fixed_vectors (torch.Tensor): Fixed vectors of shape (B, 2).
		"""
		# Compute min and max values for each sequence in the batch
		B, W, C = x.shape
		min_val, _ = x.view(B, -1).min(dim=1, keepdim=True)  # Shape: (B, 1)
		max_val, _ = x.view(B, -1).max(dim=1, keepdim=True)  # Shape: (B, 1)

		# Rescale sequences to be between 0 and 1
		scaled_sequences = (x - min_val.view(B, 1, 1)) / (max_val*(1+EPS) - min_val).view(B, 1, 1)

		# Compute the fixed vector values
		variation = (max_val - min_val) / (max_val)  # Shape: (B, 1)
		variation = torch.log(variation + 1)
		fixed_vectors = torch.cat([variation], dim=1)  # Shape: (B, 2)
		construction_vectors = torch.cat([min_val, max_val], dim=1)

		return scaled_sequences, fixed_vectors, construction_vectors


def get_min_max(construction_vectors, data = None):
	min_val = construction_vectors[:,0].view(-1, 1)
	max_val = construction_vectors[:,1].view(-1, 1)

	while data is not None and len(data.shape) > len(min_val.shape):
		min_val = min_val.unsqueeze(-1)
		max_val = max_val.unsqueeze(-1)
	return min_val, max_val

def scale_data(data, construction_vectors):
	min_val, max_val = get_min_max(construction_vectors, data=data)
	scaled = (data - min_val) / (max_val*(1+EPS)-min_val)
	return scaled

def unscale_data(data, construction_vectors):
	min_val, max_val = get_min_max(construction_vectors, data=data)
	unscaled = min_val + data * (max_val*(1+EPS)-min_val)
	return unscaled

class HyperNetwork(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim):
		super(HyperNetwork, self).__init__()
		# Define the layers for the hypernetwork
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim)  # Output weights or biases
		)
	
	def forward(self, x):
		return self.network(x)

class HyperLinear(nn.Module):
	def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim):
		super(HyperLinear, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.total_params = output_dim * (input_dim + 1)  # Total number of parameters per sample
		
		# Single hypernetwork to generate both weights and biases
		self.hypernet = HyperNetwork(embedding_dim, self.total_params, hidden_dim)
		
	def forward(self, x, task_embedding):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (B, W, input_dim).
			task_embedding (torch.Tensor): Embedding tensor of shape (B, embedding_dim).
		
		Returns:
			torch.Tensor: Output tensor after applying the hypernetwork-generated linear layer.
		"""
		B = x.size(0)
		# Generate all parameters at once
		params = self.hypernet(task_embedding)  # Shape: (B, total_params)
		
		# Split params into weights and biases
		# Weights: first output_dim * input_dim elements
		# Biases: last output_dim elements
		weights = params[:, :self.output_dim * self.input_dim]  # Shape: (B, output_dim * input_dim)
		biases = params[:, self.output_dim * self.input_dim:]    # Shape: (B, output_dim)
		
		# Reshape weights to (B, output_dim, input_dim)
		weights = weights.view(B, self.output_dim, self.input_dim)
		
		# Transpose weights for batch matrix multiplication
		weights_transposed = weights.transpose(1, 2)  # Shape: (B, input_dim, output_dim)
		
		# Perform batch-wise matrix multiplication
		# x: (B, W, input_dim)
		# weights_transposed: (B, input_dim, output_dim)
		output = torch.bmm(x, weights_transposed)  # Shape: (B, W, output_dim)
		
		# Add biases
		output = output + biases.unsqueeze(1)  # Shape: (B, W, output_dim)
		
		return output

class OHLCEmbeddingHyperNetwork(nn.Module):
	def __init__(self, input_dim, ohlc_embedding_dim, fixed_embedding_dim, hyper_net_hidden_dim):
		super(OHLCEmbeddingNetwork, self).__init__()
		self.fc1 = HyperLinear(input_dim, ohlc_embedding_dim, fixed_embedding_dim, hyper_net_hidden_dim)
		self.fc2 = HyperLinear(ohlc_embedding_dim, ohlc_embedding_dim, fixed_embedding_dim, hyper_net_hidden_dim)

	def forward(self, x, task_embedding):
		"""
		Args:
			x (torch.Tensor): Scaled sequences of shape (B, W, input_dim).
			task_embedding (torch.Tensor): task embedding of shape (B, fixed_embedding_dim).

		Returns:
			torch.Tensor: Final output values of shape (B, W, ohlc_embedding_dim).
		"""

		# Pass through HyperLinear layers
		x = F.relu(self.fc1(x, task_embedding))  # Shape: (B, W, ohlc_embedding_dim)
		x = x + self.fc2(x, task_embedding) 
		embeddings = F.tanh(x)  # Shape: (B, W, ohlc_embedding_dim)

		return embeddings

class OHLCEmbeddingNetwork(nn.Module):
	def __init__(self, input_dim, ohlc_embedding_dim):
		super(OHLCEmbeddingNetwork, self).__init__()
		hidden_dim = 2*ohlc_embedding_dim
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, ohlc_embedding_dim)

	def forward(self, x, *args):
		"""
		Args:
			x (torch.Tensor): Scaled sequences of shape (B, W, input_dim).

		Returns:
			torch.Tensor: Final output values of shape (B, 1).
		"""

		x = F.relu(self.fc1(x))
		x = x + F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))

		embeddings = F.tanh(x)

		return embeddings

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		"""
		Args:
			d_model (int): Embedding dimension.
			dropout (float): Dropout rate.
			max_len (int): Maximum length of the input sequences.
		"""
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute positional encodings once in log space
		position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

		pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
		pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

		pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (B, W, E).
		"""
		x = x + self.pe[:x.size(1)].transpose(0, 1)  # Adjust indexing and transpose
		return self.dropout(x)

class TransformerNetwork(nn.Module):
	def __init__(self, embedding_dim, max_len=1000, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
		"""
		Args:
			ohlc_embedding_dim (int): Dimension of the input embeddings.
			nhead (int): Number of heads in the multi-head attention mechanism.
			num_layers (int): Number of Transformer encoder layers.
			dim_feedforward (int): Dimension of the feedforward network model.
			dropout (float): Dropout rate.
		"""
		super(TransformerNetwork, self).__init__()
		self.embedding_dim = embedding_dim

		# Positional Encoding (optional but recommended for Transformers)
		self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout, max_len=max_len)

		# Transformer Encoder Layer
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embedding_dim,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			activation='relu',
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# Final output layer
		self.fc_out = nn.Linear(embedding_dim, 1)

		self.attn_weights = nn.Parameter(torch.Tensor(window_size))
		nn.init.uniform_(self.attn_weights, -0.1, 0.1)

	def forward(self, embeddings):
		"""
		Args:
			embeddings (torch.Tensor): Input embeddings of shape (B, W, ohlc_embedding_dim).

		Returns:
			torch.Tensor: Predictions of shape (B, 1).
		"""

		# Apply positional encoding
		embeddings = self.pos_encoder(embeddings)

		# Transformer Encoder
		transformer_output = self.transformer_encoder(embeddings)

		attn_weights = F.softmax(self.attn_weights, dim=0)  # Shape: (W,)
		attn_weights = attn_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, W, 1)
		transformer_output = transformer_output * attn_weights  # Element-wise multiplication
		transformer_output = transformer_output.sum(dim=1)  # Shape: (B, E)

		# Final output layer
		output = self.fc_out(transformer_output)  # Shape: (B, 1)

		return output


device = torch.device('cuda')
fields = ['open', 'high', 'low', 'close']
data = data_loader.load_instrument_data('BTCUSD', fields)
data = torch.tensor(data[fields].values, dtype=torch.float32, device=device)
input_dim = len(fields)
ohlc_embedding_dim = 32
fixed_embedding_dim = 1
hyper_net_hidden_dim = 32
batch_size = 2048
window_size = 128
epochs = 3
nhead = 4
num_transformer_layers = 2
transformer_dim_feedforward = ohlc_embedding_dim*2
dropout = 0.3
lr = 1e-4

dataset = TemporalDataset(data, window_size=window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

scaler_net = SequenceScaler().to(device)
ohlc_net = OHLCEmbeddingNetwork(input_dim, ohlc_embedding_dim).to(device)
main_net = TransformerNetwork(
	embedding_dim=ohlc_embedding_dim,
	max_len=window_size,
	nhead=nhead,
	num_layers=num_transformer_layers,
	dim_feedforward=transformer_dim_feedforward,
	dropout=dropout
).to(device)

criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(ohlc_net.parameters()) + list(main_net.parameters()), lr=lr)

writer = SummaryWriter(log_dir='crypto_bot/runs/')  # You can specify a directory

# sample_input = torch.randn(batch_size, window_size, input_dim).to(device)
# fixed_vectors = torch.randn(batch_size, fixed_embedding_dim).to(device)
# writer.add_graph(ohlc_net, (sample_input, fixed_vectors))

for epoch in range(epochs):
	for (idx, minibatch) in enumerate(dataloader):


		unscaled_sequences, targets = minibatch  # Unpack
		scaled_sequences, fixed_vectors, construction_vectors = scaler_net(unscaled_sequences)

		embeddings = ohlc_net(scaled_sequences, fixed_vectors)

		predictions = main_net(embeddings)
		scaled_targets = scale_data(targets, construction_vectors)
		
		loss = criterion(predictions, scaled_targets)

		# Backpropagation and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		global_step = epoch*len(dataloader) + idx+1
		if global_step%1000 == 0 or global_step==1:
			unscaled_predictions = unscale_data(predictions, construction_vectors)
			log_predictions_to_tensorboard(writer, unscaled_sequences, targets, unscaled_predictions, 4, 3, global_step)
		writer.add_scalar('Loss/train', loss.item(), global_step)

writer.close()

# from .models.transformer_model_v1 import QNetwork
# from .agents.Qagent import Qagent

# env = MarketEnv(external_config_dir='crypto_bot/configs', config_name='default')
# obs, _ = env.reset()

# print([k for k in obs.keys()])


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