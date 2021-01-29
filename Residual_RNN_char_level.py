import torch
import torch.nn as nn
import os
from Model import RNN
from Model_evaluation import sample
from Tools import data_preparation


# optimizer and model must be defined before
def train(hidden1, hidden2, hidden3, model, contex_input, contex_target):
	contex_target.unsqueeze_(-1)
	h1, h2, h3 = hidden1, hidden2, hidden3
	model.zero_grad()
	loss = 0
	
	# [context_size; hidden_size]
	for i in range(contex_input.size(0)):
		output, h1, h2, h3 = model(contex_input[i], h1, h2, h3)
		l = criterion(output, contex_target[i])
		loss += l

	loss.backward()
	optimizer.step()
	# print(loss.item())
									# general loss
	return h1.data, h2.data, h3.data, loss.item() / contex_input.size(0)

torch.manual_seed(0)
torch.set_deterministic(True)
save_folder = 'parameters/'
if len(os.listdir(path=save_folder))!=0:
	checkpoint = torch.load(save_folder+'auxiliary_data.pt')
	source, alphabet, source_lenght, \
	 vocabulary_size, letter_transform, \
	 indexes_transform, HS, CS, criterion = checkpoint['data']

	checkpoint = torch.load(save_folder+'parameters.pt')
	model = RNN(vocabulary_size, HS)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loss = checkpoint['loss']
	n = checkpoint['n']
	p = checkpoint['p']
	h1, h2, h3 = checkpoint['hidden_values']

	nn.utils.clip_grad_value_(model.parameters(), 2)
	print('Recovery completed successfully!')
else:
	source = open('input.txt', 'r').read()
	alphabet = list(set(source))
	source_lenght, vocabulary_size = len(source), len(alphabet)
	letter_transform = { letter:i for i, letter in enumerate(alphabet) }
	indexes_transform = { i:letter for i, letter in enumerate(alphabet) }
	print('vocabulary_size: ', vocabulary_size)
	HS = 128 # hidden size
	CS = 10 # Context size

	criterion = nn.CrossEntropyLoss()
	model = RNN(vocabulary_size, HS)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
	# optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-45, momentum=0.9, nesterov=True)
	nn.utils.clip_grad_value_(model.parameters(), 2)
	# nn.utils.clip_grad_norm_(model.parameters(), 2)
	loss, n, p = 100, 0, 0

	# save fields
	torch.save({
		'data': [source, alphabet, source_lenght, 
				 vocabulary_size, letter_transform, 
				 indexes_transform, HS, CS, criterion]
		}, save_folder+'auxiliary_data.pt')
model.train()

while True:
	if p+CS+1 >= source_lenght or n == 0:
		h1 = model.initHidden() # hidden_values1
		h2 = model.initHidden() # hidden_values2
		h3 = model.initHidden() # hidden_values3
		p = 0

	inputs = [letter_transform[ch] for ch in source[p:p+CS]]
	inputs = data_preparation(inputs, vocabulary_size)
	targets = [letter_transform[ch] for ch in source[p+1:p+CS+1]]
	targets = torch.LongTensor(targets)

	h1, h2, h3, loss_value = train(h1, h2, h3, model, inputs, targets)
	loss = loss * 0.99 + loss_value * 0.01

	# checking intermediate results
	if n % 500 == 0:
		'''
		Every sample generation looks like following:
		model.eval() 
		# sample generation
		model.train()
		'''
		model.eval()
		start_letter = [letter_transform[source[p]]]
		text_sample = sample(h1, h2, h3, model, vocabulary_size, start_letter, 200)
		text_sample = ''.join(indexes_transform[index] for index in text_sample)
		print ('------\n %s \n------' % (text_sample, ))
		model.train()

		# print('iter %d, iter loss: %f' % (n, loss_value))
		print('iter %d, loss: %f' % (n, loss))

	# checkpoint establishing
	if n % 2000 == 0:
		torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
			'n': n,
			'p': p,
			'hidden_values': (h1, h2, h3)
			}, save_folder+'parameters.pt')
		print('checkpoint\'s done!')

	p+=CS
	n+=1
	# print(n)
	