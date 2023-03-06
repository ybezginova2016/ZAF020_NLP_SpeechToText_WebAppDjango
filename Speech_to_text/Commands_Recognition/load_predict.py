import pandas as pd 
import os
from IPython.display import display,Audio
import cv2
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn import preprocessing


#Load the trained model
class NN2DMEL(nn.Module):
    def __init__(self, num_class):
        super(NN2DMEL,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1)
        self.dropout1 = nn.Dropout(0.3) 
    
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(768, 256)
        self.dropout5 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256,128)
        self.dropout6 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_class)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),kernel_size=3)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=3)
        x = self.dropout2(x)
        x = F.relu(self.fc1(x.reshape(-1,x.shape[1] * x.shape[2]*x.shape[3])))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        #print(x.shape)
        return x 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = NN2DMEL(num_class=6)

net = net.to(device)

state_model_dict =  torch.load(
    '/kaggle/input/speech-command-reognition-model-mel-spec/epoch_194.pth',
    map_location=torch.device('cpu')
)

net.load_state_dict(state_model_dict)

net.eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)


#Create Input tensor
class TextProcessed:
    def __init__(self,characters = None,commands = None):
        if characters != None:
            #For transcription recognize
            self.characters = characters
            self.characters_map = dict()
            self.index_characters_map = dict()
            for i, character in enumerate(self.characters):
                self.characters_map[character] = i
                self.index_characters_map[i] = character
        if commands != None:
            #for classification
            self.commands = commands
            self.commands_dict = dict()
            self.index_commands_dict = dict()
            for i, command in enumerate(self.commands):
                self.commands_dict[command] = i
                self.index_commands_dict[i] = command
                
    def text2int(self, text):
        int_list = list()
        for ch in text:
            int_list.append(self.characters_map[ch])
        return int_list
    
    def int2text(self, int_list):
        ch_list = list()
        for int_ch in int_list:
            ch_list.append(self.index_characters_map[int_ch])
        return ''.join(ch_list)
    
commands_list = ['go','stop','forward','down','left','right']
tp = TextProcessed(commands = commands_list)

waveform, sample_rate = torchaudio.load('/kaggle/input/audio-command-sample/sample.wav')

mel_transform  = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
mel_waveform = mel_transform(waveform)

input_tensor =  mel_waveform[None,None,:,:]
print(input_tensor[0].shape)


#Get output tensor.
output_tensor = net(input_tensor[0])


#Decode output tensor
print('Prediction :')
tp.index_commands_dict[int(torch.max(output_tensor.data, 1).indices)]
    
