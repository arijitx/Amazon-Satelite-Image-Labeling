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
		self.gru = nn.GRU(embed_size,hidden_size,batch_first=True)
		self.linear_cnn = nn.Linear(cnn_size,total_size)
		self.linear_gru = nn.Linear(hidden_size,total_size)
		self.linear_class = nn.Linear(total_size,n_classes)
		self.init_weights()

	def init_weights(self):
		init.xavier_uniform(self.sos_param)

	def forward(self,cnn_feat,labels,lens):
		batch_size=labels.size(0)
		embeddings = self.embed(labels)
		sos_param = self.sos_param.repeat(batch_size,1,1)
		embeddings = torch.cat([sos_param,embeddings],dim=1)
		packed = pack(embeddings,lens,batch_first=True)
		outputs,hiddens = self.gru(packed)
		unpacked = unpack(outputs,batch_first=True)[0]

		cnn_feat = cnn_feat.squeeze()
		cnn_feat_linear = self.linear_cnn(cnn_feat).unsqueeze(1)
		cnn_feat_linear = cnn_feat_linear.repeat(1,unpacked.size(1),1)

		gru_linear = self.linear_gru(unpacked)

		combined_feat = nn.functional.relu(cnn_feat_linear+gru_linear)

		final_output = self.linear_class(combined_feat)

		return final_output

	def sample(self,cnn_feat):
		batch_size = cnn_feat.size(0)
		sampled_labels = []
		current_label_embed = self.sos_param.repeat(batch_size,1,1)
		cnn_feat=cnn_feat.squeeze()
		states=None
		cnn_feat_linear=self.linear_cnn(cnn_feat).unsqueeze(1)

		for i in range(18):
			outputs,states=self.gru(current_label_embed,states)

			gru_linear=self.linear_gru(outputs)
			combined_feat=nn.functional.relu(gru_linear+cnn_feat_linear)

			final_output=self.linear_class(combined_feat)
			current_label=final_output.max(2)[1]
			current_label_embed=self.embed(current_label)
			sampled_labels.append(current_label)

		return torch.cat(sampled_labels,dim=1)





    
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, cnn_output_size, num_labels, combined_size):
#         """Set the hyper-parameters and build the layers."""
#         super(DecoderRNN, self).__init__()

#         self.start_vect = nn.Parameter(torch.zeros(1, 1, embed_size), requires_grad = True)
#         self.embed = nn.Embedding(num_labels + 1, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
#         self.linear_lstm = nn.Linear(hidden_size, combined_size)
#         self.linear_cnn = nn.Linear(cnn_output_size, combined_size)
#         self.linear_final = nn.Linear(combined_size, num_labels + 1)
#         self.init_weights()
    
#     def init_weights(self):
#         """Initialize weights."""
#         nn.init.xavier_uniform(self.start_vect)
#         nn.init.xavier_uniform(self.embed.weight)
#         nn.init.xavier_uniform(self.linear_lstm.weight)
#         self.linear_lstm.bias.data.fill_(0)
#         nn.init.xavier_uniform(self.linear_cnn.weight)
#         self.linear_cnn.bias.data.fill_(0)
#         nn.init.xavier_uniform(self.linear_final.weight)
#         self.linear_final.bias.data.fill_(0)
        
#     def forward(self, cnn_features, labels, lengths):
#         embeddings = self.embed(labels)
#         stacked_start = torch.cat([self.start_vect for _ in range(embeddings.size(0))])
#         embeddings = torch.cat((stacked_start, embeddings), 1)

#         packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
#         hiddens, _ = self.lstm(packed)
#         unpacked = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
#         unbound = torch.unbind(unpacked, 1)
#         combined = [self.linear_lstm(elem) for elem in unbound]
#         combined = torch.stack(combined, 1)
#         #cnn_features = cnn_features.squeeze()
#         projected_image = self.linear_cnn(cnn_features)
#         combined += torch.stack([projected_image for _ in range(combined.size(1))], 1)
        
#         divided = torch.unbind(nn.functional.relu(combined), 1)
#         outputs = [self.linear_final(elem) for elem in divided]

#         return torch.stack(outputs, 1)
    
#     def sample(self, cnn_features, states=None):
#         """Samples captions for given image features (Greedy search)."""
#         sampled_labels = []
#         inputs = torch.cat([self.start_vect for _ in range(cnn_features.size(0))])

#         for i in range(18):                                      # maximum sampling length
#             hiddens, states = self.lstm(inputs, states)
#             combined = self.linear_lstm(hiddens.squeeze(1))
#             combined += self.linear_cnn(cnn_features)
#             outputs = self.linear_final(nn.functional.relu(combined))
#             predicted = outputs.max(1)[1]
#             sampled_labels.append(predicted.unsqueeze(1))
#             inputs = self.embed(predicted)
#             inputs=inputs.unsqueeze(1)

#         sampled_labels = torch.cat(sampled_labels, 1)

#         return sampled_labels.squeeze()
