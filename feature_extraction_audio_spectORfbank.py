#importing modules
#general
import os
import argparse
import numpy as np
import pandas as pd
import pickle
import time
#processing
import torch
import torchaudio
from torchaudio.compliance import kaldi

# argparse constructor
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True,
	help = "path to the data file directory")
parser.add_argument("-l", "--labels", required=True,
	help = "path to the labels")
parser.add_argument("-f", "--features", required=True,
	help = "type of feature to extract (spectrogram or fbank")
args = vars(parser.parse_args())

#creating a list with paths from directory
list_of_files = sorted([os.path.join(args["data"], i) for i in os.listdir(args["data"])])

#extracting the features
def extract_features(wav, sample_rate, feature_type):
	if feature_type == "spectrogram":
		if sample_rate != 16000:
			wav = kaldi.resample_waveform(wav, sample_rate, 16000)
			sample_rate = 16000
		f = kaldi.spectrogram(wav, sample_frequency=sample_rate)
		
	elif feature_type == "fbank":
		if sample_rate != 16000:
			wav = kaldi.resample_waveform(wav, sample_rate, 16000)
			sample_rate = 16000
		f = kaldi.fbank(wav, sample_frequency=sample_rate)
	#print(f'NEW sample rate: {sample_rate}')
	
	return f
	
data = {}       
def process_audio(files, feature_type):
	start_time = time.time()
	for file_path in files:
		waveform, sample_rate = torchaudio.load_wav(file_path)
		#print(f'OG sample rate: {sample_rate}')
		#print(waveform, sample_rate)
		features = extract_features(waveform, sample_rate, feature_type)
		data[file_path.split('/')[-1].replace('.wav', '')] = features.numpy()#.tolist if storing as .json (gets rid of array type)
	end_time = time.time()
	duration = end_time - start_time
	print(f'audio processing of {len(files)} file(s) took {duration} seconds')

process_audio(list_of_files, args["features"])

#extracting the labels
ref = ["anger", "happiness", "neutral", "sadness"]
labels = {}
def get_labels(file):
	start_time = time.time()
	nr_l = 0
	nr_bl = 0
	with open(file, 'r') as f:
		for line in f:
			inner_d = {}
			splt = line.split(';')
			if splt[1] in ref: # filtering down to 4 calsses
				inner_d["LABEL"] = splt[1]
				inner_d["ACTIVATION"] = float(splt[2][2:])
				inner_d["VALENCE"] = float(splt[3][2:])
				labels[splt[0]] = inner_d
			else:
				labels["X"+splt[0]] = None # marking file-label if category not in ref
				nr_bl += 1
			nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'label processing of {nr_l} line(s) took {duration} seconds')
	print(f'nr of bad labels: {nr_bl}')
get_labels(args["labels"])

#creating a dictionary
dataset_d = {}
def create_dataset(labels, data):
	start_time = time.time()
	nr_l = 0
	for x, y, i, j  in zip(labels.keys(), data.keys(), labels.values(), data.values()):
		#print(x, y, i, j)
		if x == y:
			features = {} 
			features["FEATURES"]= j
			i.update(features)
			dataset_d[x] = i
		#else:
			#print(x[1:] == y) # file-labels are matching after the X-marking
			#print(f'filtered out: {x} {y}')
		nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'combining of {nr_l} line(s) took {duration} seconds')
	
create_dataset(labels, data)
print(f'original length of dataset was: {len(data)} VS new length of dataset is: {len(dataset_d)}')
#creating a pandas dataframe
df = pd.DataFrame.from_dict(dataset_d, orient='index')

#storing the dataframe as file
if args["features"] == "spectrogram":
	print("spectrogram")
	df.to_pickle('/mount/arbeitsdaten/thesis-dp-1/vollenia/dataframes/df_audio_spectrogram.pkl')
elif args["features"] == "fbank":
	print("fbank")
	df.to_pickle('/mount/arbeitsdaten/thesis-dp-1/vollenia/dataframes/df_audio_fbank.pkl')
