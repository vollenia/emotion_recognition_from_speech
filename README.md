# emotion_recognition_from_speech

The goal of this project is to train and evaluate a model that would be able to automatically recognize one of 4 emotion classes (Happy, Angry, Sad, Neutral) from a speech signal. Given the assumption that prosody is the main contrubutor to recognizing emotions from speech, all lexical content is disregarded in this project.

### Data

Emotion Class | Count | Percentage
------------ | ------------- | ------------- 
Total | 7,798 | 100.00%
Angry | 792 | 10.16%
Happy | 2,644 | 33.91%
Neutral | 3,477 | 44.59%
Sad | 885 | 11.35%

### Workflow

* Mel log filter banks as features (middle-ground between spectrograms with largre size / redundant information and MFCCs with a high leven of abstraction)
* Feature extraction via Kaldi extension for torchaudio
* Features padded/cut to a universal shape of 691 frames (mean nr of frames + std)
* Features combined with the labels into a pandas dataframe
* Normalization via Z-Score (mean=0; std=1)
* Model training with a CNN style architecture adapted to time continuous data. Takes a convolution over all features of one or several time steps resulting in a one-dimensional movement of the kernel which corresponds to the movement along the time dimension. Model's architecture is displayed in Figure X.
* Modles were trained and evaluated in a 6-fold cross validation approach

### Evaluation results



