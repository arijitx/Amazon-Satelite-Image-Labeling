import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable
import torch.nn.functional as F
import sys



class EncoderCNN(nn.Module):
	def __init__(self):
		super(EncoderCNN,self).__init__()
		m=models.vgg16(pretrained=True)
		layers=list(list(m.children())[0].children())[:-3]
		self.model = nn.Sequential(*layers)
		self.conv1 = nn.Conv2d(512,10,1,1)
		self.linear = nn.Linear(10*14*14,18)
		self.output_size=512

	def forward(self, images,class_label=False):
		if class_label:
			x = self.model(images)
			x = F.relu(self.conv1(x))
			x = x.view(images.size(0),10*14*14)
			x = self.linear(x)
			return x
		return self.model(images)

class Attn(nn.Module):
	def __init__(self, hidden_size):
		super(Attn, self).__init__()
		self.hidden_size = hidden_size
		self.wh = nn.Linear(hidden_size, hidden_size)
		self.wc = nn.Linear(hidden_size, hidden_size)
		self.v = nn.Linear(hidden_size,1)

	def forward(self, hidden, cnn_feats):
		# hidden [B,1,H]
		# cnn_feats [B,T,H]

		batch_size = cnn_feats.size(0) #B
		cnn_feats_len = cnn_feats.size(1) #T
		hidden = hidden.repeat(1,cnn_feats_len,1) #[B,T,H]

		energy = self.wh(hidden)+self.wc(cnn_feats) #[B,T,H]
		energy = F.tanh(energy) # [B,T,H]
		energy = self.v(energy) # [B,T,1]
		energy = F.softmax(energy,dim=1) #[B,T,1]
		return energy


class DecoderRNN(nn.Module):
	def __init__(self,embed_size,hidden_size,cnn_size,n_classes,total_size):
		super(DecoderRNN,self).__init__()
		self.sos_param = nn.Parameter(torch.zeros(1,1,embed_size),requires_grad=True)
		self.h0 = nn.Parameter(torch.zeros(1,1,hidden_size),requires_grad=True)
		self.embed = nn.Embedding(n_classes,embed_size)
		self.gru = nn.GRU(embed_size+hidden_size,hidden_size,batch_first=True)
		self.linear_cnn = nn.Linear(hidden_size,embed_size)
		self.linear_class = nn.Linear(hidden_size,n_classes)
		self.attn=Attn(hidden_size)
		self.init_weights()


	def init_weights(self):
		init.xavier_uniform(self.sos_param)
		init.xavier_uniform(self.h0)

	def forward(self,cnn_feat,labels,lens):
		batch_size=labels.size(0)
		embeddings = self.embed(labels)
		sos_param = self.sos_param.repeat(batch_size,1,1)
		embeddings = torch.cat([sos_param,embeddings],dim=1)
		cnn_feats = cnn_feat.view(batch_size,14*14,-1) # [B,T,H]

		hidden=self.h0.repeat(batch_size,1,1).transpose(0,1) # [1,B,H]

		outputs=[]

		for i in range(embeddings.size(1)-1):
			embedding = embeddings[:,i,:].unsqueeze(1) # [B,1,E]
			attn_weights = self.attn(hidden.transpose(0,1),cnn_feats) # [B,T,1]
			context = attn_weights.transpose(1,2).bmm(cnn_feats) # [B,1,H]
			embedding = torch.cat([embedding,context],dim=2) # [B,1,H+E]  
			output,hidden=self.gru(embedding,hidden) 
			outputs.append(output)

		outputs = torch.cat(outputs,dim=1)
		final_output = self.linear_class(outputs)

		return final_output

	def sample(self,cnn_feat):
		batch_size = cnn_feat.size(0)
		sampled_labels = []
		current_label_embed = self.sos_param.repeat(batch_size,1,1)
		cnn_feat=cnn_feat.view(batch_size,14*14,-1)
		states=self.h0.repeat(batch_size,1,1).transpose(0,1)
		attn_weights_list=[]

		for i in range(18):
			attn_weights = self.attn(states.transpose(0,1),cnn_feat)
			context = attn_weights.transpose(1,2).bmm(cnn_feat)

			current_label_embed=torch.cat([current_label_embed,context],dim=2)
			outputs,states=self.gru(current_label_embed,states)

			final_output=self.linear_class(nn.functional.relu(outputs))
			current_label=final_output.max(2)[1]
			current_label_embed=self.embed(current_label)
			sampled_labels.append(current_label)
			attn_weights_list.append(attn_weights.transpose(1,2))

		attn_weights_list=torch.cat(attn_weights_list,dim=1)
		return attn_weights_list,torch.cat(sampled_labels,dim=1)

