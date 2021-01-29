import torch
import torch.nn as nn


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.i2h1 = nn.Linear(input_size + hidden_size, hidden_size)
		self.act1 = nn.ReLU()
		# self.act1 = nn.Tanh()

		self.h12h2 = nn.Linear(hidden_size + hidden_size, hidden_size)
		self.act2 = nn.ReLU()
		# self.act2 = nn.Tanh()

		self.h22h3 = nn.Linear(hidden_size + hidden_size, hidden_size)
		self.act3 = nn.ReLU()
		# self.act3 = nn.Tanh()

		self.h2o = nn.Linear(hidden_size, input_size)
		# input_size = output_size, because one-hot encode was used

	def forward(self, input, hidden1, hidden2, hidden3):
		input_combined = torch.cat((input, hidden1), 1)

		inner_hidden1 = self.i2h1(input_combined)
		inner_hidden1 = self.act1(inner_hidden1)
		# inner_hidden1 += hidden1

		hidden1_combined = torch.cat((hidden2, inner_hidden1), 1)

		inner_hidden2 = self.h12h2(hidden1_combined)
		inner_hidden2 = self.act2(inner_hidden2)
		# inner_hidden2 += hidden2

		hidden2_combined = torch.cat((hidden3, inner_hidden2), 1)

		inner_hidden3 = self.h22h3(hidden2_combined)
		inner_hidden3 = self.act3(inner_hidden3)
		# inner_hidden3 += hidden3

		output = self.h2o(inner_hidden3)

		return output, inner_hidden1, inner_hidden2, inner_hidden3

	def initHidden(self):
		# We put only 1 token per moment: [1; hidden_size]
		return torch.zeros(1, self.hidden_size)