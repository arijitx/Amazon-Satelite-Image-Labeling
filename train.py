# torch imports
import torch
from torch import nn
from torch.autograd import Variable as var 
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torchvision.transforms as T
from torch.utils.data import DataLoader

# standard imports
import json
import pandas as pd
from collections import namedtuple
import argparse
from AmazonDataset import AmazonDataset
import os
import numpy as np
# from models.cnn_rnn_caption import EncoderCNN,DecoderRNN
from models.cnn_rnn_attn import EncoderCNN, DecoderRNN


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def f2_score(pred,true_labels,eps = 1e-8):
	pos = len(pred & true_labels)
	precision = 1.0 * pos / (len(pred) + eps)
	recall = 1.0 * pos / (len(true_labels) + eps)
	beta = 2
	f2 = (1.0 + beta**2)*precision*recall / (beta**2 * precision + recall + eps)
	return f2

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda(config.training.cuda_device)
	return var(x, volatile=volatile)

def parse():
	config=json.loads(open('config.json','r').read(),object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
	return config

def collate_fn(data):
	# manipulate batch
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, labels = zip(*data)
	images = torch.stack(images, 0)
	lengths = [len(label) for label in labels]
	targets = torch.zeros(len(labels), max(lengths)).long()
	for i, label in enumerate(labels):
		end = lengths[i]
		targets[i, :end] = label[:end]
	return images, targets, lengths

def main(config):
	best_f2=0

	def add_eos(target):
		return torch.Tensor(sorted(target) + [18]*(18 - len(target)))

	def one_hot(target):
		target=[x-1 for x in target]
		target_oh=np.zeros(18)
		target_oh[target]=1.
		return torch.Tensor(target_oh)

	loss_list=[]

	# train dataset transform
	train_transform = T.Compose([
		T.Resize(256),
		T.RandomResizedCrop(224),
		T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
	])

	# val dataset transform
	val_transform = T.Compose([
		T.Resize(224),
		T.CenterCrop(224),
		T.ToTensor(),
		T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])

	# init train Dataset
	train_dataset = AmazonDataset(config.paths.train_dir, 
							config.paths.train_labels_file, 
							config.paths.label_list_file, 
							transform = train_transform, 
							target_transform = add_eos)

	# init train Dataset
	train_dataset_pretraining = AmazonDataset(config.paths.train_dir, 
							config.paths.train_labels_file, 
							config.paths.label_list_file, 
							transform = train_transform, 
							target_transform = one_hot)

	# init val Dataset
	val_dataset = AmazonDataset(config.paths.val_dir, 
							config.paths.val_labels_file, 
							config.paths.label_list_file,
							transform = val_transform,
							target_transform = add_eos)

	# init train loader
	train_loader = DataLoader(train_dataset,
							batch_size=config.training.batch_size,
							num_workers=config.training.n_workers,
							collate_fn = collate_fn,
							shuffle=True)

	train_loader_pretraining = DataLoader(train_dataset_pretraining,
							batch_size=config.training.batch_size,
							num_workers=config.training.n_workers,
							shuffle=True)
	# init val loader
	val_loader = DataLoader(val_dataset, 
							batch_size = config.training.batch_size,
							num_workers = config.training.n_workers)

	# Build the models
	encoder = EncoderCNN()
	decoder = DecoderRNN(config.model.embed_size, config.model.hidden_size, encoder.output_size, 19, config.model.total_size)

	# convert models to cuda if cuda available
	if torch.cuda.is_available():
		encoder.cuda(config.training.cuda_device)
		decoder.cuda(config.training.cuda_device)
	#####################################
	###          Pretraining          ###
	#####################################
	train_pretrain_params=list(encoder.parameters())
	lr_1=config.training.lr_2
	bce_loss=nn.BCEWithLogitsLoss()
	for i in range(5):
		print("Epoch :",(i+1),"/",5)
		optimizer=torch.optim.Adam(train_pretrain_params,lr=lr_1)
		for idx,(images,labels) in enumerate(train_loader_pretraining):
			images = to_var(images)
			labels = to_var(labels)

			encoder.zero_grad()
			decoder.zero_grad()

			logits = encoder.forward(images,class_label=True)

			loss=bce_loss(logits,labels)
			loss.backward()

			optimizer.step()

			if (idx+1)%config.training.n_batch_print==0:
				print("Batch [%d/%d] Loss : %.4f"%((idx+1),len(train_loader_pretraining),loss.data[0]))

		lr_1=lr_1/config.training.lr_2_decay

	#####################################
	###         RNN Training          ###
	#####################################

	train_1_params=list(decoder.parameters())
	lr_1=config.training.lr_1
	cec_loss=nn.CrossEntropyLoss()

	decoder.train()

	# decoder training
	for i in range(config.training.n_epoch_1):
		print("Epoch :",(i+1),"/",config.training.n_epoch_1)
		optimizer=torch.optim.Adam(train_1_params,lr=lr_1)
		# training loop
		for idx,(images,labels,lengths) in enumerate(train_loader):
			images=to_var(images)
			labels=to_var(labels)

			encoder.zero_grad()
			decoder.zero_grad()

			cnn_features=encoder(images)
			outputs=decoder(cnn_features,labels,lengths)

			unbound_labels=torch.unbind(labels,1)
			unbound_outputs=torch.unbind(outputs,1)

			losses=[cec_loss(unbound_outputs[j],unbound_labels[j]) for j in range(len(unbound_labels))]
			loss=sum(losses)
			
			loss.backward()

			optimizer.step()
			
			loss_list.append(loss.data[0])
			if (idx+1)%config.training.n_batch_print==0:
				print("Batch [%d/%d] Loss : %.4f"%((idx+1),len(train_loader),loss.data[0]))
				
		lr_1=lr_1/config.training.lr_1_decay

		# Validation
		encoder.eval()
		decoder.eval()

		f2_score_total=0.
		n_samples=0

		for k, (images, labels) in enumerate(val_loader):
			images=to_var(images,volatile=True)
			cnn_features=encoder(images)

			attn,preds=decoder.sample(cnn_features)
			batch_size=images.size(0)
			for j in range(batch_size):
				pred=preds[j].data.cpu().numpy().tolist()
				if 18 in pred:
					pred=pred[:pred.index(18)]

				pred=set(pred)
				true_labels=set(labels[j])
				true_labels.remove(18)

				f2_score_total+=f2_score(pred,true_labels)
				n_samples+=1

		f2=f2_score_total/n_samples

		print("F2 Score :",f2)
		if f2 > best_f2:
			best_f2 = f2
			print('found a new best!')
			torch.save(decoder.state_dict(), config.paths.rnn_save_path)
			torch.save(encoder.state_dict(), config.paths.cnn_save_path)

	#####################################
	### CNN finetuning+ RNN training  ###
	#####################################
	encoder.train()
	decoder.train()

	train_2_params=list(decoder.parameters())+list(encoder.parameters())
	lr_2=config.training.lr_2
	cec_loss=nn.CrossEntropyLoss()


	for i in range(config.training.n_epoch_2):
		print("Epoch :",(i+1),"/",config.training.n_epoch_2)
		optimizer=torch.optim.Adam(train_2_params,lr=lr_2)
		# training loop
		for idx,(images,labels,lengths) in enumerate(train_loader):
			images=to_var(images)
			labels=to_var(labels)

			encoder.zero_grad()
			decoder.zero_grad()

			cnn_features=encoder(images)
			outputs=decoder(cnn_features,labels,lengths)

			unbound_labels=torch.unbind(labels,1)
			unbound_outputs=torch.unbind(outputs,1)

			losses=[cec_loss(unbound_outputs[j],unbound_labels[j]) for j in range(len(unbound_labels))]
			loss=sum(losses)
			
			loss.backward()

			optimizer.step()
			
			loss_list.append(loss.data[0])
			if (idx+1)%config.training.n_batch_print==0:
				print("Batch [%d/%d] Loss : %.4f"%((idx+1),len(train_loader),loss.data[0]))

		lr_2=lr_2/config.training.lr_2_decay

		# Validation
		encoder.eval()
		decoder.eval()

		f2_score_total=0.
		n_samples=0

		for k, (images, labels) in enumerate(val_loader):
			images=to_var(images,volatile=True)
			cnn_features=encoder(images)

			attn,preds=decoder.sample(cnn_features)
			batch_size=images.size(0)
			for j in range(batch_size):
				pred=preds[j].data.cpu().numpy().tolist()
				if 18 in pred:
					pred=pred[:pred.index(18)]

				pred=set(pred)
				true_labels=set(labels[j])
				true_labels.remove(18)

				f2_score_total+=f2_score(pred,true_labels)
				n_samples+=1

		f2=f2_score_total/n_samples

		print("F2 Score :",f2)
		if f2 > best_f2:
			best_f2 = f2
			print('found a new best!')
			torch.save(decoder.state_dict(), config.paths.rnn_save_path)
			torch.save(encoder.state_dict(), config.paths.cnn_save_path)

		index=list(range(len(loss_list)))

		loss_track=pd.DataFrame()
		loss_track['index']=index
		loss_track['loss']=loss_list
		loss_track.to_csv(config.paths.save_loss_path,index=False)

if __name__=='__main__':
	config=parse()
	main(config)
