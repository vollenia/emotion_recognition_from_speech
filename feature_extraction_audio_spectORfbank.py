# Importing modules
import os
import argparse
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torchaudio
from torchaudio.compliance import kaldi

# Argparse constructor
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True,
	help = "path to the data file directory")
parser.add_argument('-l', '--labels', required=True,
	help = "path to the labels")
parser.add_argument('-f', '--features', required=True,
	help = "type of feature to extract (spectrogram or fbank")
args = vars(parser.parse_args())

# Extracting features
def extract_features(wav, sample_rate, feature_type):
	if feature_type == 'spectrogram':
		if sample_rate != 16000:
			wav = kaldi.resample_waveform(wav, sample_rate, 16000)
			sample_rate = 16000
		f = kaldi.spectrogram(wav, sample_frequency=sample_rate)	
	elif feature_type == 'fbank':
		if sample_rate != 16000:
			wav = kaldi.resample_waveform(wav, sample_rate, 16000)
			sample_rate = 16000
		f = kaldi.fbank(wav, sample_frequency=sample_rate)
	return f
	
def process_audio(files, feature_type):
	start_time = time.time()
	data = {}
	for file_path in files:
		waveform, sample_rate = torchaudio.load(file_path)
		features = extract_features(waveform, sample_rate, feature_type)
		# note: use '\\' to spit if running on Windows
		data[file_path.split('/')[-1].replace('.wav', '')] = features.numpy() # .tolist if storing as .json (gets rid of array type)
	end_time = time.time()
	duration = end_time - start_time
	print(f'audio processing of {len(files)} file(s) took {duration} seconds')
	return data

# Extracting labels
def get_labels(file):
	start_time = time.time()
	ref = ['anger', 'happiness', 'neutral', 'sadness']
	labels = {}
	nr_l = 0
	nr_bl = 0
	with open(file, 'r') as f:
		for line in f:
			inner_d = {}
			splt = line.split(';')
			if splt[1] in ref: # filtering down to 4 calsses
				inner_d['LABEL'] = splt[1]
				inner_d['ACTIVATION'] = float(splt[2][2:])
				inner_d['VALENCE'] = float(splt[3][2:])
				labels[splt[0]] = inner_d
			else:
				labels['X'+splt[0]] = None # marking file-label if category not in ref
				nr_bl += 1
			nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'label processing of {nr_l} line(s) took {duration} seconds')
	print(f'nr of bad labels: {nr_bl}')
	return labels

# Creating dataset as dictionary
def create_dataset(labels, data):
	start_time = time.time()
	dataset_d = {}
	nr_l = 0
	for x, y, i, j  in zip(labels.keys(), data.keys(), labels.values(), data.values()):
		if x == y:
			features = {} 
			features['FEATURES']= j
			i.update(features)
			dataset_d[x] = i
		nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'combining of {nr_l} line(s) took {duration} seconds')
	return dataset_d
	
if __name__ == '__main__':
	# Creating a list with paths from directory
	list_of_files = sorted([os.path.join(args['data'], i) for i in os.listdir(args['data'])])
	
	# Extracting features
	data = process_audio(list_of_files, args["features"])
	
	# Extracting labels
	ref = ['anger', 'happiness', 'neutral', 'sadness']
	labels = get_labels(args['labels'])

	# Creating dataset as dictionary
	dataset_d = create_dataset(labels, data)
	print(f'original length of dataset was: {len(data)} VS new length of dataset is: {len(dataset_d)}')
	
	# Creating pandas dataframe from dictionary
	df = pd.DataFrame.from_dict(dataset_d, orient='index')
	
	# Storing the dataframe
	df.to_pickle("df_audio_" + str(args["features"]) +".pkl")