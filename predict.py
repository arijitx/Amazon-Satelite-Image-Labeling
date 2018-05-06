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
import numpy as np
from collections import namedtuple
from AmazonDataset import AmazonDataset
import os
import sys
# from models.cnn_rnn import EncoderCNN,DecoderRNN
from models.cnn_rnn_attn import EncoderCNN,DecoderRNN
from utils import draw_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse():
	config_file=sys.argv[1]
	config=json.loads(open(config_file,'r').read(),object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
	return config

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda(config.training.cuda_device)
	return var(x, volatile=volatile)

def find_classes(label_list_file):
	classes=[]
	with open(label_list_file) as f:
		for line in f:
			classes.append(line.strip)
			classes=np.array(classes)
			return classes

def filename_clean(target):
	return target.split('.')[0]

def main(config):
	test_transform = T.Compose([
		T.Resize(224),
		T.CenterCrop(224),
		T.ToTensor(),
		T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])

	test_dataset = 	AmazonDataset(config.paths.test_dir,
							transform=test_transform,
							target_transform=filename_clean,
							dtype="test")

	test_loader = DataLoader(test_dataset,
							batch_size = config.training.batch_size,
							num_workers = config.training.n_workers)

	# define model
	encoder = EncoderCNN()
	decoder = DecoderRNN(config.model.embed_size, config.model.hidden_size, encoder.output_size, 19, config.model.total_size)

	encoder.load_state_dict(torch.load(config.paths.cnn_save_path))
	decoder.load_state_dict(torch.load(config.paths.rnn_save_path))

	if torch.cuda.is_available():
		encoder.cuda(config.training.cuda_device)
		decoder.cuda(config.training.cuda_device)

	encoder.eval()
	decoder.eval()

	filenames=[]
	predictions=[]

	classes=find_classes(config.paths.label_list_file)

	print("Running Predictions :")
	for i,(images,filename) in enumerate(test_loader):
		print("Batch [%d/%d]"%((i+1),len(test_loader)))

		images=to_var(images,volatile=True)
		cnn_features=encoder(images)
		if attention:
			attn,preds=decoder.sample(cnn_features)
		else:
			preds=decoder.sample(cnn_features)
		prediction=[]
		for j in range(preds.size(0)):
			pred=preds[j].data.cpu().numpy().tolist()
			if 18 in pred:
				pred=pred[:pred.index(18)]

			prediction.append(' '.join([classes[k-1] for k in pred]))

		if attention and config.training.draw_image:
			draw_image(attn,filename,prediction)
		filenames+=list(filename)
		predictions+=prediction

	submission=pd.DataFrame()
	submission['image_name']=filenames
	submission['tags']=predictions
	submission.to_csv(config.paths.submission_file,index=False)




if __name__=='__main__':
	config=parse()
	attention = config.training.attention
	main(config)