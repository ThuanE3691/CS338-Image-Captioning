import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', None)
from tqdm.auto import tqdm
import pickle 
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os



class extractFeatureEfficientNetV2():
  def __init__(self,data):
    self.data = data
    self.scaler = transforms.Resize([224, 224])
    self.normalizer = transforms.Normalize(
        mean  = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    self.transform = transforms.ToTensor()
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    img_name = self.data.iloc[idx]['image']
    img = Image.open(img_name).convert('RGB')
    img =  self.normalizer(self.transform((self.scaler(img))))
    if img.shape[0] == 1:
      print(img.shape)
    return img_name, img

class fine_tune_model(nn.Module):
    def __init__(self, model_img_layer_4):
        super(fine_tune_model, self).__init__()
        self.model_img_layer_4 = model_img_layer_4
        self.conv = nn.LazyConv2d(512, 1, padding='same')
    def forward(self, x):
        x = self.model_img_layer_4(x)
        x = self.conv(x)
        return x
    
@st.cache_resource
def init():
  # The batch size is set to 64, meaning that 64 samples of data will be processed in one forward/backward pass of the network during training.
  batch_size = 64

  # The embedding size is set to 1280, meaning that each input sample will be represented by a vector of size 1280.
  embedding_size = 1280

  max_sequence_len = 35


  word_dict = pickle.load(open('./app/word.dict','rb'))
  word_to_idx = {word:idx for (idx,word) in enumerate(word_dict)}
  idx_to_word = {idx:word for (idx,word) in enumerate(word_dict)}
  vocab_size = len(word_to_idx)
  start = word_to_idx['<start>']
  end = word_to_idx['<end>']
  pad = word_to_idx['<pad>']

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model_img = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
  model_img.eval()
  model_img_layer_4 = model_img._modules.get('features')
  EfficientNet_V2_layer_4 = fine_tune_model(model_img_layer_4).to(device)
  return batch_size, embedding_size, max_sequence_len, word_dict, word_to_idx, idx_to_word, vocab_size, start, end, pad,device, model_img, model_img_layer_4, EfficientNet_V2_layer_4 
    
batch_size, embedding_size, max_sequence_len, word_dict, word_to_idx, idx_to_word, vocab_size, start, end, pad,device, model_img, model_img_layer_4, EfficientNet_V2_layer_4 = init()

def get_vector(t_img):
  my_emb = torch.zeros(1, embedding_size, 7, 7)
  t_img = torch.autograd.Variable(t_img).to(device)
  def hook(model, input, output):

    my_emb.copy_(output.data)
  h = model_img_layer_4.register_forward_hook(hook)
  model_img_layer_4(t_img)
  h.remove()
  return my_emb


def gen_caption(k , image_name, valid_img_emb, model_inference):

  max_sequence_len = globals()['max_sequence_len']

  img = Image.open(image_name)

  plt.imshow(img)
  img_emb = valid_img_emb[image_name] # 1 512 7 7 
  img_emb = img_emb.permute(0,2,3,1) # 1 7 7 512
  img_emb = img_emb.reshape(img_emb.size(0), -1, img_emb.size(3))
  caption = []
  seq = [pad] * max_sequence_len 
  seq[0] = start
  seq = torch.tensor(seq).squeeze(0).view(1, -1).to(device)
  img_emb = img_emb.to(device)
  
  for i in range(0,max_sequence_len-1):
    out , _ = model_inference(seq, img_emb) # 33, 32, 10000
    pred = out[i, 0, :]
    indicies = torch.topk(pred , k ).indices.tolist()
    values = torch.topk(pred, k).values.tolist()
    token = random.choices(indicies, values)[0] # 
    seq[:, i+1] = token
    if token == pad:
      break
    word = idx_to_word[token]
    caption.append(word)
  return caption




class position_encoding(nn.Module):
  def __init__(self,d_model = 512, max_len = max_sequence_len, dropout = 0.1):
    super().__init__()

    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(max_len, d_model) # (33,512)
    pos = torch.arange(0,max_len).unsqueeze(1) # (33,1)
    div_term = torch.exp(torch.arange(0,d_model, 2 ).float() * (-math.log(10000.0) / d_model)) # 256
    pe[:,::2] = torch.sin(pos * div_term)
    pe[:,1::2] = torch.cos(pos * div_term)
    pe = pe.unsqueeze(0) # (1,32,512)
    self.register_buffer('pe', pe)

  def forward(self,x):
    if x.size(0) > self.pe.size(0):
      self.pe = self.pe.repeat(x.size(0), 1, 1)
    self.pe = self.pe[:x.size(0), :, :]
    return self.dropout(self.pe +x)
  
class Imagecaptionmodel(nn.Module):
  def __init__(self, vocab_size=vocab_size, embedding_size=embedding_size, max_len=max_sequence_len, n_head=16, num_decoder_layer=4):
    super().__init__()
    self.position_encoding = position_encoding(d_model = embedding_size, max_len = max_len)

    self.transformer_decoder_layer  = nn.TransformerDecoderLayer(d_model = embedding_size, nhead = n_head)
    self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers = num_decoder_layer)
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.FC = nn.LazyLinear(vocab_size)
    self.initweights()
    self.embedding_size = embedding_size
  def initweights(self):
    self.embedding.weight.data.uniform_(-0.1, 0.1)
    self.FC.weight.data.uniform_(-0.1, 0.1)
    self.FC.bias.data.zero_()
  def create_mask(self, seq):
      'create mask for mask attention'
      attention_mask  = torch.ones(seq.size(1), seq.size(1))
      # print('attention_mask ',attention_mask.shape)
      attention_mask  = torch.tril(attention_mask)
      attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, 0)

      pad_mask = seq.masked_fill(seq == 0, float(0.0)).masked_fill(seq > 0, float(1.0))
      pad_mask_bool = seq == 0
      return attention_mask, pad_mask, pad_mask_bool
  def forward(self,seq, image_embedding):
    image_embedding  = image_embedding.permute(1,0,2) # 49,32,512

    x = self.embedding(seq) * math.sqrt(self.embedding_size)
    x = self.position_encoding(x) # 32, 33 512
    x = x.permute(1, 0, 2) # (seqlen, batchsize, embedding)

    attention_mask, pad_mask, pad_mask_bool = self.create_mask(seq)
    attention_mask, pad_mask, pad_mask_bool = attention_mask.to(device), pad_mask.to(device), pad_mask_bool.to(device)

    x = self.transformer_decoder(memory = image_embedding, tgt = x, tgt_mask = attention_mask, tgt_key_padding_mask = pad_mask_bool
                                 ) 
    out = self.FC(x)
    return out, pad_mask
  

def predict(model_inference):
  list_temp = {'image':['image.png']}
  demo = pd.DataFrame(list_temp)
  img_extract = extractFeatureEfficientNetV2(demo)
  img_loader = DataLoader(img_extract, batch_size = 1, shuffle = False)
  img_emb_test1 = {name[0]:get_vector(img) for name, img in tqdm(img_loader)}

  with open('valid_img_emb1.pkl', 'wb') as f:
    pickle.dump(img_emb_test1, f)

  valid_img_emb = pd.read_pickle('valid_img_emb1.pkl')
  result = ' '.join(gen_caption(1, demo['image'].iloc[0], valid_img_emb,model_inference)[:-1])
  display_result = f'<p style="font-size:25px;">Caption for the Image: {result}</p>'
  st.markdown(display_result, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def download_model_file():
    model_file_url = "https://drive.google.com/u/0/uc?id=1LZT6WihXPpXQ5DddR1PlSZ15w0YchZYc"
    output = "model.model"
    gdown.download(model_file_url, output, quiet=False)

@st.cache_resource
def load_model():
    model_inference = torch.load('app/model.model',map_location=torch.device('cpu'))
    return model_inference

@st.cache_resource
def download_model():
    if st.button("Download Model"):
      with st.spinner("Downloading model..."):
          download_model_file()
      st.success("Model downloaded successfully!")

def main(model_inference):
    
    st.title("Image Uploader")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        Image.open(uploaded_file).save('image.png')
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        with st.spinner('Waiting for generate caption for image'):
          predict(model_inference)



def is_model_available():
    return os.path.isfile('./app/model.model')

def check_model():
  model = None
  if is_model_available():
    st.success("Model is available!")
    model = load_model()
  else:
    st.error("Model is not available. Please download it.")
    if st.button("Download Model"):
        download_model_file()
        st.success("Model downloaded successfully!")
        model = load_model()
  return model


model_inference = None
with st.spinner('Please hold on for a moment while we verify and prepare the model to assist you'):
  if is_model_available():
    st.success("Model is available!")
    model_inference = load_model()
  else:
    st.error("Model is not available. Please download it.")
    if st.button("Download Model"):
        download_model_file()
        st.success("Model downloaded successfully!")
        model_inference = load_model()

if model_inference:
  main(model_inference)

