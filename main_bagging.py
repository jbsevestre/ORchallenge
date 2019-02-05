from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import argparse
from tqdm import tqdm
import PIL.Image as Image

import shutil

plt.ion()



########################################################################



### PART 1 : MODEL CONSTRUCTION



########################################################################

### nb : it is necessary to create a "blank_images" folder in the "bird_dataset" folder where the bird class files are empty (without images).
### you must also rename the "train_images" file to "shuffle_images".

list_of_class = ['004.Groove_billed_Ani','009.Brewer_Blackbird','010.Red_winged_Blackbird',
'011.Rusty_Blackbird','012.Yellow_headed_Blackbird', '013.Bobolink', '014.Indigo_Bunting',
'015.Lazuli_Bunting', '016.Painted_Bunting', '019.Gray_Catbird', '020.Yellow_breasted_Chat',
'021.Eastern_Towhee', '023.Brandt_Cormorant', '026.Bronzed_Cowbird', '028.Brown_Creeper',
'029.American_Crow', '030.Fish_Crow', '031.Black_billed_Cuckoo', '033.Yellow_billed_Cuckoo',
'034.Gray_crowned_Rosy_Finch']

### we create 10 models
size_of_train = 1082
nb_of_predictors = 10
for n in range(0,nb_of_predictors):

	### we do BAGGING
	shutil.copytree('bird_dataset/blank_images', 'bird_dataset/train_images_'+str(n+1))
	### for the TRAIN
	for j in range(0,size_of_train):
		r1 = np.random.randint(0,len(list_of_class))
		list_of_images_of_the_class = os.listdir('bird_dataset/shuffle_images/'+str(list_of_class[r1]))
		r2 = np.random.randint(0,len(list_of_images_of_the_class))
		image_choose = list_of_images_of_the_class[r2]
		shutil.copyfile('bird_dataset/shuffle_images/'+str(list_of_class[r1])+'/'+list_of_images_of_the_class[r2],
		 	'bird_dataset/train_images_'+str(n+1)+'/'+str(list_of_class[r1])+'/'+os.path.splitext(list_of_images_of_the_class[r2])[0]+'_'+str(j+1)+os.path.splitext(list_of_images_of_the_class[r2])[1]) ### we add a small code so that two same images work
	#####################
	
	data_dir = 'bird_dataset'
	TRAIN = 'train_images_'+str(n+1)
	VAL = 'val_images'

	data_transforms = {
		TRAIN: transforms.Compose([
			transforms.Resize(256),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		]),
		VAL: transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	}
	
	image_datasets = {
		x: datasets.ImageFolder(
			os.path.join(data_dir, x), 
			transform=data_transforms[x]
		)
		for x in [TRAIN, VAL]
	}
	
	dataloaders = {
		x: torch.utils.data.DataLoader(
			image_datasets[x], batch_size=8,
			shuffle=True, num_workers=4
		)
		for x in [TRAIN]
	}
	
	val_loader = {
		x: torch.utils.data.DataLoader(
			image_datasets[x], batch_size=8,
			shuffle=True, num_workers=4
		)
		for x in [VAL]
	}

	######################

	dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}
	class_names = image_datasets[TRAIN].classes
	
	########################################################################

	rsn152 = models.resnet152(pretrained=True)
	lt=8
	cntr=0
	for child in rsn152.children():
		cntr+=1
		if cntr < lt:
			for param in child.parameters():
				param.requires_grad = False
	num_ftrs = rsn152.fc.in_features
	rsn152.fc = nn.Linear(num_ftrs, 20)

	########################################################################

	criterion = nn.CrossEntropyLoss()
	optimizer_ft = optim.SGD(rsn152.parameters(), lr=0.001, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	########################################################################

	def train_model(current_model, criterion, optimizer, scheduler, num_epochs=10):
		since = time.time()
		best_model_wts = copy.deepcopy(current_model.state_dict())
		best_acc = 0.0
		avg_loss = 0
		avg_acc = 0
		avg_loss_val = 0
		avg_acc_val = 0	
		train_batches = len(dataloaders[TRAIN])
		val_batches = len(val_loader[VAL])	
		for epoch in range(num_epochs):
			loss_train = 0
			loss_val = 0
			acc_train = 0
			acc_val = 0
			#############################################
			current_model.train(True)
			for i, data in enumerate(dataloaders[TRAIN]):
				inputs, labels = data
				with torch.no_grad():			
					inputs, labels = Variable(inputs), Variable(labels)
				optimizer.zero_grad()
				outputs = current_model(inputs)
				_,preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				loss_train += loss.item()
				acc_train += torch.sum(preds == labels.data)
				if i==(dataset_sizes[TRAIN] // 8):
					divisor = divisor + (dataset_sizes[TRAIN] % 8)
				else:
					divisor = (i+1)*8
				print("MODEL : {}/{} - EPOCH : {}/{} - DATA : TRAIN - BATCH : {}/{:.0f} - AVERAGE LOSS : {:.4f} - ACCURACY : {}/{:.0f} ({:.4f}%)".format(n+1, nb_of_predictors, epoch+1, num_epochs, i+1, train_batches, loss_train / divisor, acc_train, divisor, (acc_train * 100 / divisor)))		
				del inputs, labels, outputs, preds
			print('----------')
			avg_loss = loss_train / dataset_sizes[TRAIN]
			avg_acc = 100 * acc_train / dataset_sizes[TRAIN]
			current_model.train(False)
			#############################################
			current_model.eval()
			for i, data in enumerate(val_loader[VAL]):
				inputs, labels = data
				with torch.no_grad():
				     inputs, labels = Variable(inputs), Variable(labels)
				optimizer.zero_grad()
				outputs = current_model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				loss_val += loss.item()
				acc_val += torch.sum(preds == labels.data)
				if i==(dataset_sizes[VAL] // 8):
					divisor = divisor + (dataset_sizes[VAL] % 8)
				else:
					divisor = (i+1)*8
				print("MODEL : {}/{} - EPOCH : {}/{} - DATA : VALIDATION - BATCH : {}/{:.0f} - AVERAGE LOSS : {:.4f} - ACCURACY : {}/{:.0f} ({:.4f}%)".format(n+1, nb_of_predictors, epoch+1, num_epochs, i+1, val_batches , loss_val / divisor, acc_val, divisor, (acc_val * 100 / divisor)))
				del inputs, labels, outputs, preds
			print('----------')
			avg_loss_val = loss_val / dataset_sizes[VAL]
			avg_acc_val = 100*acc_val / dataset_sizes[VAL]
			#############################################
			print("Epoch {} result: ".format(epoch+1))
			print("Train Average Loss : {:.4f}".format(avg_loss))
			print("Train Accuracy : {:.4f}".format(avg_acc))
			print("Validation Average Loss : {:.4f}".format(avg_loss_val))
			print("Validation Accuracy : {:.4f}".format(avg_acc_val))
			print('----------')
			if avg_acc_val > best_acc:
				best_acc = avg_acc_val
				best_model_wts = copy.deepcopy(current_model.state_dict())
		elapsed_time = time.time() - since
		print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
		print("Best Accuracy: {:.4f}".format(best_acc))
		current_model.load_state_dict(best_model_wts)
		return current_model
		
	########################################################################

	rsn152 = train_model(rsn152, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
	torch.save(rsn152.state_dict(), 'RSN152_bagging_'+str(n+1)+'on'+str(nb_of_predictors)+'.pth')
	print('--------------------')



########################################################################
	
	
	
### PART 2 : MODEL VOTE



########################################################################



data_dir = 'bird_dataset'
VAL = 'val_images'

data_transforms = {
	VAL: transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
}

image_datasets = {
	x: datasets.ImageFolder(
		os.path.join(data_dir, x), 
		transform=data_transforms[x]
	)
	for x in [VAL]
}

val_loader = {
	x: torch.utils.data.DataLoader(
		image_datasets[x], batch_size=8,
		shuffle=True, num_workers=4
	)
	for x in [VAL]
}

######################

nb_de_modeles = nb_of_predictors
model = []
for n in range(0,nb_of_predictors):
	model.append(models.resnet152())
	num_ftrs = model[n].fc.in_features
	model[n].fc = nn.Linear(num_ftrs, 20)
	state_dict = torch.load('RSN152_bagging_'+str(n+1)+'on'+str(nb_of_predictors)+'.pth')
	model[n].load_state_dict(state_dict)
	model[n].eval()
	
softy = nn.Softmax(dim=1)
accuracy = 0
total = 0

for data in image_datasets[VAL]:
	
	voting_list = np.zeros(20).astype(int)
	weighted_voting_list = np.zeros(20)
	
	inputs, label = data
	
	inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))
	
	for i in range(0,nb_de_modeles):
		output = model[i](inputs)
		_, pred_model = torch.max(output.data, 1)
		pred_model = np.asarray(pred_model)[0]
		softy_model = np.asarray(softy(output.data))[0]
		voting_list[pred_model] = voting_list[pred_model] + 1
		#weighted_voting_list[pred_model] = weighted_voting_list[pred_model] + softy_model[pred_model]**2
		
	pred = np.argmax(voting_list)
	#pred = np.argmax(weighted_voting_list)
	
	if pred == label:
		accuracy = accuracy + 1
	
	print('Predicted final class : {}'.format(pred+1))
	print('Real Class : {}'.format(label+1))
	
	print("ACCURACY after vote : {}/{:.0f} ({:.4f}%)".format(accuracy, (total+1), (accuracy * 100 / (total+1))))	
	total = total + 1
	print("------------------------------------------")
########################################################################