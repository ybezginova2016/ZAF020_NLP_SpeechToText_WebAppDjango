from transformers import AutoModelForCTC, Wav2Vec2Processor
#from IPython.display import display
import IPython.display
import torchaudio

# Load the trained model
model = AutoModelForCTC.from_pretrained("/kaggle/input/transmod/model")
processor = Wav2Vec2Processor.from_pretrained("/kaggle/input/transmod/model")
# Create Input tensor
input_tensor, sample_rate = torchaudio.load('/kaggle/input/transample/sample.wav')
# Get output tensor
with torch.no_grad():
    output_tensor = model(input_tensor).logits
# Decode output tensor
with torch.no_grad():
    pred_ids = torch.argmax(output_tensor, dim=-1)
    decode_result = processor.batch_decode(pred_ids)[0].replace("[PAD]",'')