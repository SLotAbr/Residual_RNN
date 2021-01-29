import torch
from Tools import data_preparation


'''
plans:
- generation mode: 'argmax', 'pavel'
- context_input with custom length
'''
def sample(h1, h2, h3, model, vocabulary_size, context_input, message_length):
	with torch.no_grad():
		# We should preprocess context_input here
		'''
		1) Encode context_input corresponding to embedding during training
		2) if len(context_input)>1 then we should start loop for hidden_state
			preparation (model works only with 1 token per time), i.e.
			simple loop
		'''
		# only one index now!
		# context_output = []
		# context_output.append(context_input[0])
		# context_input = data_preparation(context_input, vocabulary_size)
		context_output = []
		context_output.extend(context_input)
		if len(context_input)==1:
			context_input = data_preparation(context_input, vocabulary_size)


		# When I add the possibility of sending context_input more than 1 char,
		# message_length -1 will transform into message_length - len(context_input)
		for i in range(message_length - 1):
			output, h1, h2, h3 = model(context_input[0], h1, h2, h3)
			# print(output)

			topv, topi = output.topk(1)
			topi = topi[0][0]
			# now topi: tensor with 1 digit
			topi = int(topi)
			context_output.append(topi)
			context_input = data_preparation([topi], vocabulary_size)

		return context_output