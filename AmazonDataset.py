import torch.utils.data as data
from PIL import Image
import os
import os.path
import pandas as pd
import json

def get_classes(label_list_file):
	#
	# list of classes
	# class to idx dict
	#

	f = open(label_list_file)
	classes=[]
	class_to_idx={}
	i=1
	for line in f:
		classes.append(line.strip())
		class_to_idx[line.strip()]=i+1
		i+=1
	f.close()
	
	return classes,class_to_idx

def get_im_to_labels(labels,class_to_idx):
	#
	# image_name to labels array
	# {train_1234:[1,4,3,6], ... }
	#

	labels=json.loads(pd.read_csv(labels).to_json(orient='records'))
	im_to_labels={}
	for item in labels:
		im_to_labels[item['image_name']]=[class_to_idx[tag] for tag in item['tags'].split(' ')]

	return im_to_labels

def make_dataset(im_dir,labels_csv,class_to_idx,train=True):
	#
	# images list of tupples
	# tupple (full_path_to_image,labels_array)
	#
	if train:
		im_to_labels=get_im_to_labels(labels_csv,class_to_idx)
	images=[]

	for im_file in os.listdir(im_dir):
		im_file_path=os.path.join(im_dir,im_file)
		if train:
			images.append((im_file_path,im_to_labels[im_file.split('.')[0]]))
		else:
			images.append((im_file_path,im_file))

	return images

def load_im(path):
	with open(path, 'rb') as imf:
		with Image.open(imf) as im:
			return im.convert('RGB')

class AmazonDataset(data.Dataset):
	def __init__(self,data_dir,labels_file=None,label_list_file=None,transform=None,target_transform=None,dtype='train'):
		if dtype=="train":
			classes, class_to_idx = get_classes(label_list_file)
			imgs = make_dataset(data_dir, labels_file, class_to_idx)
		else:
			classes,class_to_idx=None,None
			imgs = make_dataset(data_dir,None,None,train=False)

		if len(imgs) == 0:
			print('No Images in directory')

		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.dtype=dtype

	def __getitem__(self,index):
		path, target = self.imgs[index]
		img=load_im(path)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def __len__(self):
		return len(self.imgs)