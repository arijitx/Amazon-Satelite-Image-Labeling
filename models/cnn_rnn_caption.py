import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable
import sys



class EncoderCNN(nn.Module):
	def __init__(self):
		super(EncoderCNN,self).__init__()
		m=models.resnet18(pretrained=True)
		layers=list(m.children())[:-1]
		self.model=nn.Sequential(*layers)
		self.output_size=512

	def forward(self, images):
		return self.model(images)

class DecoderRNN(nn.Module):
	def __init__(self,embed_size,hidden_size,cnn_size,n_classes,total_size):
		super(DecoderRNN,self).__init__()
		self.sos_param = nn.Parameter(torch.zeros(1,1,embed_size),requires_grad=True)
		self.embed = nn.Embedding(n_classes,embed_size)
		self.gru = nn.GRU(2*embed_size,hidden_size,batch_first=True)
		self.linear_cnn = nn.Linear(cnn_size,embed_size)
		self.linear_class = nn.Linear(hidden_size,n_classes)
		self.init_weights()

	def init_weights(self):
		init.xavier_uniform(self.sos_param)

	def forward(self,cnn_feat,labels,lens):
		batch_size=labels.size(0)
		embeddings = self.embed(labels)
		sos_param = self.sos_param.repeat(batch_size,1,1)
		embeddings = torch.cat([sos_param,embeddings],dim=1)

		cnn_feat = cnn_feat.squeeze()
		cnn_feat_linear = self.linear_cnn(cnn_feat).unsqueeze(1)
		cnn_feat_linear = cnn_feat_linear.repeat(1,embeddings.size(1),1)

		embeddings = torch.cat([embeddings,cnn_feat_linear],dim=2)

		packed = pack(embeddings,lens,batch_first=True)
		outputs,hiddens = self.gru(packed)
		unpacked = unpack(outputs,batch_first=True)[0]

		final_output=self.linear_class(nn.functional.relu(unpacked))

		return final_output

	def sample(self,cnn_feat):
		batch_size = cnn_feat.size(0)
		sampled_labels = []
		current_label_embed = self.sos_param.repeat(batch_size,1,1)
		cnn_feat=cnn_feat.squeeze()
		states=None
		cnn_feat_linear=self.linear_cnn(cnn_feat).unsqueeze(1)

		for i in range(18):
			current_label_embed=torch.cat([current_label_embed,cnn_feat_linear],dim=2)
			outputs,states=self.gru(current_label_embed,states)

			final_output=self.linear_class(nn.functional.relu(outputs))
			current_label=final_output.max(2)[1]
			current_label_embed=self.embed(current_label)
			sampled_labels.append(current_label)

		return torch.cat(sampled_labels,dim=1)

