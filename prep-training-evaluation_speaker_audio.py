# Importing modules
import numpy as np
import pandas as pd
import pickle
import argparse
import time
import json
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

# Batch size and epochs
BATCH_SIZE = 128
EPOCHS = 20

# MODEL(s) + SUPPORT FUNCTIONS FOR TRAINING
# CNN with Conv2D to MaxPool1D
class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 128, kernel_size=(10, 23), stride=3)
		self.drop0 = nn.Dropout(p=0.7)
		self.drop1 = nn.Dropout(p=0.7)
		#---- trying to find out what the input shape to fc1 is ----
		x = torch.randn(max_len_features, 23).view(-1, 1, max_len_features, 23) #random tensor of correct shape
		self._to_linear = None
		self.convs(x)
		#-----------------------------------------------------------
		self.fc1 = nn.Linear(self._to_linear, 128)
		self.fc2 = nn.Linear(128, 4)
		
	def convs(self, x):
		x = F.relu(self.conv1(x)) 
		x = torch.squeeze(x, dim=3) # gets rid of the last dimension (torch.Size([a, b, c, 1]) -> torch.Size([a, b, c]))
		x = F.max_pool1d(x, 5)
		x = self.drop0(x)
		if self._to_linear is None:
			self._to_linear = np.prod(x[0].shape) # catching the output (x[0].shape[0]*x[0].shape[1]*x[0].shape[2])
		return x
		
	def forward(self, x):
		x = self.convs(x) # passing through convolutional layers
		x = x.view(-1, self._to_linear) # flattening
		x = F.relu(self.fc1(x))
		x = self.drop1(x)
		x = self.fc2(x)
		return x

# This trains the model
def training(X_train, y_train):
	net.train()
	train_batch_loss = []
	train_correct = 0
	train_total = 0
	for i in range(0, len(X_train), BATCH_SIZE):
		X_train_batch = X_train[i:i+BATCH_SIZE].view(-1, 1, max_len_features, 23)
		y_train_batch = y_train[i:i+BATCH_SIZE]
		# Fitment (zeroing the gradients)
		optimizer.zero_grad()
		train_outputs = net(X_train_batch)
		for j, k in zip(train_outputs, y_train_batch):
			if torch.argmax(j) == k:
				train_correct += 1
			train_total += 1
		train_loss = loss_function(train_outputs, y_train_batch)
		train_batch_loss.append(train_loss.item())
		
		train_loss.backward()
		optimizer.step()
		
	train_loss_epoch = round(float(np.mean(train_batch_loss)),4) # over all BATCHES
	train_acc_total = round(train_correct/train_total, 4) # over all FILES
	
	return train_loss_epoch, train_acc_total

# This tests the model (dev/final_test)
def testing(X, y, final_test=False):
	net.eval()
	correct = 0
	total = 0
	batch_loss = []
	final_test_predictions = []
	with torch.no_grad():
		for i in range(0, len(X), BATCH_SIZE):
			X_batch = X[i:i+BATCH_SIZE].view(-1, 1, max_len_features, 23)
			y_batch = y[i:i+BATCH_SIZE]
			outputs = net(X_batch)
			if final_test:
				final_test_predictions.append(torch.argmax(outputs, dim=1))
			loss = loss_function(outputs, y_batch)
			batch_loss.append(loss.item())
			for j, k in zip(outputs, y_batch):
				if torch.argmax(j) == k:
					correct += 1
				total += 1
	loss_epoch = round(float(np.mean(batch_loss)),4) # over all BATCHES
	acc_total = round(correct/total, 4) # over all FILES
	
	if final_test:
		return torch.cat(final_test_predictions), acc_total
	else:
		return loss_epoch, acc_total

# Argparse constructor
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True,
	help = "path to the dataset file")
parser.add_argument("-l", "--label", required=True,
	help = "type of label to be used (LABEL or ACTIVATION or VALENCE")
args = vars(parser.parse_args())

# Loading the pandas dataset
df = pd.read_pickle(args["data"])

# Converting continuous labels (activation/valence) into discrete classes
def map_to_bin(cont_label):
	if cont_label <= 2.5:
		return 0.0
	elif 2.5 < cont_label < 3.5:
		return 1.0
	elif cont_label >= 3.5:
		return 2.0

for i in enumerate(df.index):
	df.at[i[1], 'ACTIVATION'] = map_to_bin(df['ACTIVATION'][i[0]])
	df.at[i[1], 'VALENCE'] = map_to_bin(df['VALENCE'][i[0]])

# Converting emotion labels into classes
df["LABEL"].replace({'anger': 0, 'happiness': 1, 'neutral': 2, 'sadness': 3}, inplace=True)

# Computing the maximal length of frames (mean_length + std)
nr_of_frames = [i.shape[0] for i in df["FEATURES"]]
max_len_features = int(np.mean(nr_of_frames)+np.std(nr_of_frames))
print(f'features will be cut/extended to length: {max_len_features}')

# Splitting data according to the 6 original sessions (given the speaker id)
def create_sessions(df):
	#features
	f_1 = []
	f_2 = []
	f_3 = []
	f_4 = []
	f_5 = []
	f_6 = []
	#labels (category/activation/valnece depending on the parsed argument)
	l_1 = []
	l_2 = []
	l_3 = []
	l_4 = []
	l_5 = []
	l_6 = []
	
	for i in df.index:
		session = i[17:19] # contains session nr
		if session == "01":
			f_1.append(df.loc[i,"FEATURES"])
			l_1.append(df.loc[i,args["label"]])
		elif session == "02":
			f_2.append(df.loc[i,"FEATURES"])
			l_2.append(df.loc[i,args["label"]])
		elif session == "03":
			f_3.append(df.loc[i,"FEATURES"])
			l_3.append(df.loc[i,args["label"]])
		elif session == "04":
			f_4.append(df.loc[i,"FEATURES"])
			l_4.append(df.loc[i,args["label"]])
		elif session == "05":
			f_5.append(df.loc[i,"FEATURES"])
			l_5.append(df.loc[i,args["label"]])
		elif session == "06":
			f_6.append(df.loc[i,"FEATURES"])
			l_6.append(df.loc[i,args["label"]])
		else:
			print(f'ERROR occured for: {i}')
	
	return [f_1, f_2, f_3, f_4, f_5, f_6], [l_1, l_2, l_3, l_4, l_5, l_6]

f, l = create_sessions(df)

#SUPPORT FUNCTIONS FOR THE K-FOLD-SPLIT (SESSION-WISE-SPLIT)

# Concatenating with zeros/cutting to length (TAKES: list of ndarrays CONVERTS TO: tensor)
def zeros(d, m):
	start_time = time.time()
	f = torch.stack( # concat along a NEW dim
		[torch.cat( # concatenation along one of GIVEN dims
		[
			torch.Tensor(i[:m]), # takes SLICE of i from 0 to given NR
			torch.zeros((
				(m - i.shape[0]) if m > i.shape[0] else 0, i.shape[1])) # i.shape[1] can be set to a constant since: nr of extracted features
		], dim=0)
			for i in d], dim=0)
	end_time = time.time()
	duration = end_time - start_time
	print(f'processing took {duration} seconds')
	return f

# Extracts mean and std over all the data
def mean_std(features):
	c_features = np.concatenate(features, axis=0)
	features_mean, features_std = np.mean(c_features), np.std(c_features, ddof=0)
	
	return features_mean, features_std

# Standardization (mean=0; std=1)
def standardize(features, mean, std):
	features = (features - mean) / (std + 0.0000001) # adding epsilon to avoid errors (e.g. division by 0)
	
	return features

# Splitting validation set into dev and test (least populated class needs to have AT LEAST 2 MEMBERS)
def SSS(X_val, y_val):
	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
	sss.get_n_splits(X_val, y_val)
	for train_index, test_index in sss.split(X_val, y_val):
		X_dev, X_final_test = X_val[train_index], X_val[test_index]
		y_dev, y_final_test = y_val[train_index], y_val[test_index]
		
		return X_dev, X_final_test, y_dev, y_final_test

# Creates a list of arrays (since each array has a different shape due to audio length)
def list_arrays(x):
	listed = []
	for i in x:
		for j in i:
			listed.append(j)
	
	return listed

# Provides insights into the results
def insight(actual, pred):
	total = 0
	actual_dict = {0:0, 1:0, 2:0, 3:0}
	pred_dict = {0:0, 1:0, 2:0, 3:0}
	correct_dict = {0:0, 1:0, 2:0, 3:0} 
	for a, p in zip(actual, pred):
		actual_dict[int(a)] +=1
		pred_dict[int(p)] +=1
		if a == p:
			correct_dict[int(p)] +=1
		total += 1
	print("ACTUAL:")
	for i in actual_dict:
		print(i, '\t', actual_dict[i], '\t', round(actual_dict[i]/total*100, 4), '%')
	print("PREDICTED:")
	for i in pred_dict:
		print(i, '\t', pred_dict[i], '\t', round(pred_dict[i]/total*100, 4), '%')
	print("CORRECT:")
	for i in correct_dict:
				if actual_dict[i] == 0:
						print(0)
				else:
						print(i, '\t', correct_dict[i], '\t', round(correct_dict[i]/actual_dict[i]*100, 4), '%')

# 6-FOLD CROSS VALIDATION
statistics = {}
accuracy_of_k_fold = []
F1u_of_k_fold = []
F1w_of_k_fold = []
for i in range(len(l)):
	print(f'Iteration: {i}')
	X_train, X_test = list_arrays(f[:i] + f[i+1:]), f[i]
	y_train, y_test = np.concatenate((l[:i] + l[i+1:]),axis=0), np.array(l[i])
	class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
	print(f'CLASS WEIGHTS: {class_weights}')
	# Concatenating all timeframes into one array to extract mean+std
	# (necessary since files had different length -> different nr of frames)
	features_mean, features_std = mean_std(X_train)
	# Standardizing the features of the sessions individually
	X_train = [standardize(i, features_mean, features_std) for i in X_train]
	# Cutting/padding TRAIN to uniform length
	X_train = zeros(X_train, max_len_features)
	# Standardizing TEST with MEAN & STD of TRAIN
	X_test = [standardize(i, features_mean, features_std) for i in X_test]
	# Cutting/padding TEST to uniform length
	X_test = zeros(X_test, max_len_features)   
	# Splitting TEST into DEV and FINAL_TEST
	X_dev, X_final_test, y_dev, y_final_test = SSS(X_test, y_test)
	
	# Gathering general information
	l_t, c_t = np.unique(y_train, return_counts=True)
	l_d , c_d = np.unique(y_dev, return_counts=True)
	l_f_t, c_f_t = np.unique(y_final_test, return_counts=True)
	
	general = {"train_total": len(y_train), "train_dist": c_t.tolist(), 
			"dev__total": len(y_dev), "dev__dist": c_d.tolist(), 
			"final_test__total": len(y_final_test), "final_test__dist": c_f_t.tolist()}
	
	# Converting labels and class_weights to tensors
	y_train = torch.LongTensor(y_train)
	y_dev = torch.LongTensor(y_dev).cuda()
	y_final_test = torch.LongTensor(y_final_test).cuda()
	class_weights = torch.Tensor(class_weights).cuda()
	
	X_dev = X_dev.cuda()
	X_final_test = X_final_test.cuda()
	
	# Performing random permutation
	perm_ind = torch.randperm(len(y_train))
	X_train = X_train[perm_ind].cuda()
	y_train = y_train[perm_ind].cuda()
	
	print(f' X_train shape is: {X_train.shape} y_train length is: {len(y_train)}')
	print(f' X_dev shape is: {X_dev.shape} y_dev length is: {len(y_dev)}')
	print(f' X_final_test shape is: {X_final_test.shape} y_final_test length is: {len(y_final_test)}')
	
	#-----------TRAINING STEP--------------
	net = CNN().cuda() # reinitializing the NN for each new fold (in order to get rid of the learned parameters)
	
	optimizer = optim.Adam(net.parameters(), lr=0.0001)
	loss_function = nn.CrossEntropyLoss(weight=class_weights)
	
	fold = {"general": general, "train_loss_fold": [], "train_acc_fold": [], "dev_loss_fold": [], "dev_acc_fold": []}
	for epoch in range(EPOCHS):
		# Training
		train_loss_epoch, train_acc_epoch = training(X_train, y_train)
		fold["train_loss_fold"].append(train_loss_epoch)
		fold["train_acc_fold"].append(train_acc_epoch)

		# Evaluation on DEV
		dev_loss_epoch, dev_acc_epoch = testing(X_dev, y_dev)
		fold["dev_loss_fold"].append(dev_loss_epoch)
		fold["dev_acc_fold"].append(dev_acc_epoch)
		print(f'loss: {train_loss_epoch} {dev_loss_epoch} acc: {train_acc_epoch} {dev_acc_epoch}')
	
	# Evaluation on FINAL_TEST
	final_test_predictions, final_test_acc_total = testing(X_final_test, y_final_test, final_test=True)
	fold["ACC"] = final_test_acc_total
	accuracy_of_k_fold.append(final_test_acc_total)
	print(f'Accuracy of the final test: {final_test_acc_total}%')
	
	F1u = round(f1_score(torch.clone(y_final_test).cpu(), torch.clone(final_test_predictions).cpu(), average='macro'),4)
	fold["F1u"] = F1u
	F1u_of_k_fold.append(F1u)
	print(f'F1u-Score of the final test: {F1u}')
	
	F1w = round(f1_score(torch.clone(y_final_test).cpu(), torch.clone(final_test_predictions).cpu(), average='weighted'),4)
	fold["F1w"] = F1w
	F1w_of_k_fold.append(F1w)
	print(f'F1w-Score of the final test: {F1w}')
	
	fold["y_final_test"] = y_final_test.cpu().tolist()
	fold["final_test_predictions"] = final_test_predictions.cpu().tolist()
	
	print(y_final_test[:20])
	print(final_test_predictions[:20])
	insight(y_final_test, final_test_predictions)
	statistics[i] = fold
	print('\n')

statistics["total_ACC"] = round(np.mean(accuracy_of_k_fold),4)
statistics["total_F1u"] = round(np.mean(F1u_of_k_fold),4)
statistics["toal_F1w"] = round(np.mean(F1w_of_k_fold),4)
statistics["max_len_audio"] = max_len_features
statistics["batch_size"] = BATCH_SIZE 
statistics["epochs"] = EPOCHS 

print(f'AVERAGE ACCURACY OVER FOLDS IS: {round(np.mean(accuracy_of_k_fold),4)}%')
print(f'AVERAGE F1u OVER FOLDS IS: {round(np.mean(F1u_of_k_fold),4)}')
print(f'AVERAGE F1w OVER FOLDS IS: {round(np.mean(F1w_of_k_fold),4)}')

aff = input("store the data (y/n): ")
if aff == "y":
	with open('stats_audio_'+str(args["label"])+'_t'+'.json', 'w', encoding='utf-8') as f:
		json.dump(statistics, f, ensure_ascii=False, indent=2)
