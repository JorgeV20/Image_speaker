import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from PIL import Image
from datasets import load_dataset
import soundfile as sf
import random
import string
import time
#--- IMAGE CAPTION-
@st.cache_resource
def model():

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

llm_model=model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = llm_model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

##----TEXT TO SPEECH

# load the processor
@st.cache_resource
def load_processor():
   processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
   return processor
processor = load_processor()
# load the model
@st.cache_resource
def load_speech_model():
    speech_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    return speech_model
speech_model=load_speech_model()
# load the vocoder, that is the voice 
@st.cache_resource
def load_vocoder():
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    return vocoder
vocoder=load_vocoder()
# we load this dataset to get the speaker embeddings
@st.cache_resource
def load_embeddings_dataset():
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    return embeddings_dataset
embeddings_dataset = load_embeddings_dataset()

# speaker ids from the embeddings dataset
speakers = {
    'awb': 0,     # Scottish male
    'bdl': 1138,  # US male
    'clb': 2271,  # US female
    'jmk': 3403,  # Canadian male
    'ksp': 4535,  # Indian male
    'rms': 5667,  # US male
    'slt': 6799   # US female
}


def save_text_to_speech(text, speaker=None):
    # preprocess text
    inputs = processor(text=text, return_tensors="pt").to(device)
    if speaker is not None:
        # load xvector containing speaker's voice characteristics from a dataset
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0).to(device)
    else:
        # random vector, meaning a random voice
        speaker_embeddings = torch.randn((1, 512)).to(device)
    # generate speech with the models
    speech = speech_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    if speaker is not None:
        # if we have a speaker, we use the speaker's ID in the filename
        output_filename = f"{speaker}-{'-'.join(text.split()[:6])}.mp3"
    else:
        # if we don't have a speaker, we use a random string in the filename
        random_str = ''.join(random.sample(string.ascii_letters+string.digits, k=5))
        output_filename = f"{random_str}-{'-'.join(text.split()[:6])}.mp3"
    # save the generated speech to a file with 16KHz sampling rate
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
    # return the filename for reference
    return output_filename



#---FRONT END

st.title('Image Speaker')
with st.sidebar:
    
    st.write("You can upload the image you want to describe")
    uploaded_file =st.file_uploader('Image upload here')

col1, col2 = st.columns(2)

with col1:
    if uploaded_file!=None:
        st.image(uploaded_file,  width=300)
        pred=predict_step([uploaded_file])
    #st.write(pred[0])
    #output_filename = save_text_to_speech(pred[0], speaker=speakers["slt"])
    #st.audio(output_filename, format="audio/mp3", loop=False)
    else:
       st.write('You image will appear here')

with col2:
   if uploaded_file!=None:
    #st.image(uploaded_file,  width=500)
    #pred=predict_step([uploaded_file])
    st.write('The description of the images is: ')
    st.write(pred[0])
    output_filename = save_text_to_speech(pred[0], speaker=speakers["slt"])
    st.write('You can listen the description too: ')
    st.audio(output_filename, format="audio/mp3", loop=False)