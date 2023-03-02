import json
from random import randint
from time import sleep
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
import struct
from transformers import AutoModelForCTC, Wav2Vec2Processor
from omegaconf import OmegaConf
from lib_stt.src.silero.utils import (init_jit_model, 
                       split_into_batches,
                       read_audio,
                       read_batch,
                       prepare_model_input)

from channels.generic.websocket import WebsocketConsumer



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

class NN2DMEL(nn.Module):
    def __init__(self, num_class):
        super(NN2DMEL,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1)
        self.dropout1 = nn.Dropout(0.3) 
    
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.dropout2 = nn.Dropout(0.3)
        
        #self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1)
        #self.dropout3 = nn.Dropout(0.3)
        
        #self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
        #self.dropout4 = nn.Dropout(0.3)
        
        #self.fc0 = nn.Linear(1664, 256)
        
        self.fc1 = nn.Linear(768, 2048)
        self.dropout5 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(2048,128)
        self.dropout6 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),kernel_size=3)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=3)
        x = self.dropout2(x)
        
        #x = F.max_pool2d(F.relu(self.conv3(x)),kernel_size=3)
        #x = self.dropout3(x)
        #print(x.shape)
        #x = F.max_pool2d(F.relu(self.conv4(x)),kernel_size=3)
        #x = self.dropout4(x)
        #print(x.shape)
        #print(x.shape)
        #print(x.shape)
        x = F.relu(self.fc1(x.reshape(-1,x.shape[1] * x.shape[2]*x.shape[3])))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        #print(x.shape)
        return x 

class WSConsumerCommands(WebsocketConsumer):
    #def __init__(self):
    #    self.frame_id = list()
    def connect(self):

        self.accept()

        self.commands_list = ['go','stop','up','down','backward','left','right','noise']

        self.tp = TextProcessed(commands = self.commands_list)

        self.net = NN2DMEL(num_class=8)
        self.net.load_state_dict(torch.load(
                #'commands_model_epoch_194.pth',
                'commands_model/epoch_108.pth',
                map_location=torch.device('cpu')

            )
        )
        self.net.eval()

        self.mfcc_tranform = torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=64)
        self.mel_transform  = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

    def receive(self, text_data=None, bytes_data=None):

        #first_sample_rate = 44100
        first_sample_rate = 48000

        signal_list = [struct.unpack("f",bytes_data[index*4:index*4+4])[0] for index in range(first_sample_rate)]
        #print(text_data)
        
        waveform = torch.tensor(signal_list)

        resampler = T.Resample(first_sample_rate, 16000, dtype=waveform.dtype)
        waveform = resampler(waveform)
        print('WAVEFORM SHAPE: ', waveform.shape)

        #mfcc = self.mfcc_transform(waveform)
        #print('MFCC SHAPE: ', mfcc.shape)
        mel = self.mel_transform(waveform)
        input_tensor =  mel[None,None,:,:]
        print('INPUT TENSOR SHAPE: ', input_tensor.shape)
        out = self.net(input_tensor)
        print(out)

        predicted = torch.max(out.data, 1)

        decode = self.tp.index_commands_dict[int(predicted.indices)]
        print(decode)
        self.send(json.dumps({
            'message': 'decode_callback',
            'command': decode,
            }
        ))

class WSConsumerTransformer(WebsocketConsumer):
    def connect(self):

        self.accept()

        self.send(json.dumps({
            'message': 'loading_model',
        }))

        self.signal_full_list = list()

        SAMPLING_RATE = 16000

        self.start_val = 0

        torch.set_num_threads(1)

        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            onnx=False
        )

        (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = vad_utils

        self.vad_iterator = VADIterator(vad_model)

        self.device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
        models = OmegaConf.load('lib_stt/models.yml')
        self.lib_model, self.lib_decoder = init_jit_model(models.stt_models.en.latest.jit, device=self.device)

        #self.transformer_model = AutoModelForCTC.from_pretrained("transformer/")
        #self.transformer_processor = Wav2Vec2Processor.from_pretrained("transformer/")
        self.transformer_model = AutoModelForCTC.from_pretrained("transformer_K/")
        self.transformer_processor = Wav2Vec2Processor.from_pretrained("transformer_K/")

        self.send(json.dumps({
            'message': 'model_is_ready',
        }))

        self.lib = True


    def receive(self, text_data=None, bytes_data=None):

        signal_chunk_list = [struct.unpack("f",bytes_data[index*4:index*4+4])[0] for index in range(4608)]
        #self.signal_full_list += signal_chunk_list

        waveform = torch.tensor(signal_chunk_list)
        resampler = T.Resample(48000, 16000, dtype=waveform.dtype)
        waveform = resampler(waveform)
        self.signal_full_list += waveform.tolist()

        speech_dict = self.vad_iterator(waveform) #, return_seconds=True)

        #print(waveform)
        #print(speech_dict)

        if speech_dict != None:
            #print(list(speech_dict.keys()))
            if list(speech_dict.keys())[0] == 'start':
                #print(speech_dict['start'])
                self.start_val = speech_dict['start']
            elif list(speech_dict.keys())[0] == 'end':
                #print(speech_dict['end'])
                end_val = speech_dict['end']
                sequence_waveform_list = self.signal_full_list[self.start_val:end_val]
                sequence_waveform_tensor = torch.tensor(sequence_waveform_list).unsqueeze(0)
                #print(sequence_waveform_tensor)
                #print(sequence_waveform_tensor.shape)
                if self.lib == False:
                    with torch.no_grad():
                        logits = self.transformer_model(sequence_waveform_tensor).logits
                        pred_ids = torch.argmax(logits, dim=-1)
                        decode_result = self.transformer_processor.batch_decode(pred_ids)[0].replace("[PAD]",'')
                        print(decode_result)
                        self.send(json.dumps({
                        'message': 'decoded_result',
                        'decoded_result': decode_result,
                        }))
                else:
                    output = self.lib_model(sequence_waveform_tensor)
                    for example in output:
                        decode_result = self.lib_decoder(example.cpu())
                        print(decode_result)
                        self.send(json.dumps({
                        'message': 'decoded_result',
                        'decoded_result': decode_result,
                        }))

                    


class ConsumerClass(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send(text_data = json.dumps({
            'message': 'hi'
        }))
    #def receive(self, text_data = None, byte_data = None):
    #    pass