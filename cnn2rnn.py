import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import CrossEntropyLoss

import numpy as np 

import load_data


class EncoderCNN(nn.Module):
	def __init__(self,embeding_size,CUDA_DEVICE):
		super(EncoderCNN,self).__init__()
		m=models.resnet152(pretrained=True)
		layers=list(m.children())[:-1]
		self.cnn=nn.Sequential(*layers).cuda(CUDA_DEVICE)
		self.linear=nn.Linear(m.fc.in_features,embeding_size).cuda(CUDA_DEVICE)
		self.linear.weight.data.normal_(0.0, 0.02)
		self.linear.bias.data.fill_(0)
		self.bn = nn.BatchNorm1d(embeding_size, momentum=0.01).cuda(CUDA_DEVICE)

	def forward(self,inputs):
		features = self.cnn(inputs)
		features = Variable(features.data)
		features = features.view(features.size(0), -1)
		features = self.linear(features)
		features = self.bn(features)
		return features

class DecoderRNN(nn.Module):
	def __init__(self,embeding_size,hidden_size,output_size,CUDA_DEVICE):
		super(DecoderRNN,self).__init__()
		self.embed = nn.Embedding(output_size, embeding_size).cuda(CUDA_DEVICE)
		self.gru=nn.GRU(embeding_size,hidden_size,batch_first=True).cuda(CUDA_DEVICE)
		self.linear=nn.Linear(hidden_size,output_size).cuda(CUDA_DEVICE)
		self.softmax=nn.Softmax(dim=1)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)
		self.embed.weight.data.uniform_(-0.1, 0.1)

	def forward(self,inputs,start_state,lengths):
		inputs=self.embed(inputs)	
		start_state=start_state.unsqueeze(1)
		new_inputs=torch.cat([start_state, inputs], dim=1)
		lengths=lengths.cpu().data.numpy().flatten().astype('int').tolist()
		packed = pack_padded_sequence(new_inputs, lengths, batch_first=True) 
		hiddens, _ = self.gru(packed)
		outputs = self.linear(hiddens[0])
		return outputs

	def infer(self,start_state,max_samples):
		eos_reached=False
		EOS_TOKEN=18

		context=start_state
		context=context.unsqueeze(1)
		predictions=[]
		outputs=[]
		samples_count=0

		states=None
		while(samples_count<max_samples):
			hiddens,states=self.gru(context,states)
			output=self.linear(hiddens)
			predicted = output.max(2)[1]
			context=self.embed(predicted)
			# print(predicted)
			predictions.append(predicted.data)
			outputs.append(output)
			samples_count+=1


		return torch.cat(predictions,1),torch.cat(outputs,1)






# if __name__=='__main__':
# 	# images=load_data.load_images('test_ims.npy')
# 	# im_batch=Variable(torch.from_numpy(images).float())

# 	# enc=EncoderCNN(1000)
# 	trains,targets,lens=load_data.load_labels('data/labels.npy')
# 	# trains=load_data.one_hot_labels(trains[:10],20)
# 	# targets=load_data.one_hot_labels(targets[:10],20)
# 	# lens=load_data.one_hot_labels(lens[:10],20)
# 	trains=trains[:10].astype('int')
# 	targets=targets[:10].astype('int')
# 	lens=lens[:10].astype('int')

# 	print(lens)
# 	print(trains)
# 	dec=DecoderRNN(1000,256,20).double()

# 	# print(torch.from_numpy(images))
# 	# context=enc.forward(im_batch)
# 	# print(np.array(context).shape)
# 	# np.save('context.npy',context.data.numpy())
# 	context=np.load('context.npy')
# 	context=Variable(torch.from_numpy(context)).double()
# 	output=dec.forward(trains,context,lens)
# 	loss=CrossEntropyLoss()
# 	lengths=lens.flatten().astype('int').tolist()
# 	trains = pack_padded_sequence(Variable(torch.from_numpy(trains)), lengths, batch_first=True)[0]
# 	print(loss(output,trains))