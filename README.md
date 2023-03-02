# Speech-To-Text Django Web Application
## Demonstration
### Home Page
#### The html document of this page also includes a script that establishes a socket connection to the backend and processes input commands. The response of the page html is the change in the states of the button containers.
![](https://github.com/Begelit/Speech-To-Text-Web-App-Django/blob/main/Django_Web_Application/demo/Home_Trim.gif)
### Commands Recognition Page
#### This page exists to demonstrate the ability to control web elements in real time and the ability to visualize the input sound wave.
![](https://github.com/Begelit/Speech-To-Text-Web-App-Django/blob/main/Django_Web_Application/demo/Commands_Trim.gif)
### Transformer Speech-To-Text
#### This page tells about the result of the pre-trained transformer model trained by us.
![](https://github.com/Begelit/Speech-To-Text-Web-App-Django/blob/main/Django_Web_Application/demo/Transformer_Trim.gif)
### Ready-made Speech-To-Text model.
#### We also tried to deploy a library model of speech recognition in our application - Silero-VAD.
![](https://github.com/Begelit/Speech-To-Text-Web-App-Django/blob/main/Django_Web_Application/demo/readymade_Trim.gif)
## Models
### for a description of the process of training models, you can go to this repository with Jupyter Notebooks.
#### https://github.com/Begelit/Speech-To-Text-Training
#### P.S. Thanks to Kevin(Kaggle - kevinvaishnav) for participating in the project, teaching Transformer model and providing the necessary resources!
#### https://www.kaggle.com/code/kevinvaishnav/stt-readymade/notebook
### Sources
#### Commands Recognition Model
##### For successfully use the model that recognizes your commands, you need to download [this](https://drive.google.com/drive/folders/18H3d_jIhHubffwwYof1yWygiyKPTQkvK?usp=sharing) file and copy it to this directory "./Speech-To-Text-Web-App-Django/Django_Web_Application/commands_model/".
#### Transformer Model
##### Fir use transformer model in this web application you need download "pytorch_model.bin" file from [this](https://drive.google.com/drive/folders/1uI2faf1LFaHuI2DbuvQj4M7sdiLYktIw?usp=sharing) link and copy it to this directory "./Speech-To-Text-Web-App-Django/Django_Web_Application/transformer_model/"
