import torch


# one-hot encode
def data_preparation(context_indices, vocabulary_size):
	data_length = len(context_indices)
	# Extra dimension for convenient picking any token in input data
	data = torch.zeros(data_length, 1, vocabulary_size)
	for l in range(data_length):
		data[l][0][context_indices[l]] = 1
	# [context_size, 1, vocabulary_size]
	return data