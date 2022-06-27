#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -----------------------------------------------------------
#Vec2Node: Self-training with Tensor Augmentation for Text Classification with Few Labels
# Authors: Sara Abdali, Subhabrata Mukherjee, Evangelos Papalexakis
# (C) ECML-PKDD 2022
# sabda005@ucr.edu
# -----------------------------------------------------------
from transformers import MarianMTModel, MarianTokenizer
import torch
use_cuda = torch.cuda.is_available()
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch_device)
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

def download(model_name):
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name).to(torch_device)
  return tokenizer, model

tmp_lang_tokenizer, tmp_lang_model = download('Helsinki-NLP/opus-mt-en-ROMANCE')
src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-ROMANCE-en')
def translate(texts, model, tokenizer, language):
  """Translate texts into a target language"""
  formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
  original_texts = [formatter_fn(txt) for txt in texts]
  tokens = tokenizer.prepare_translation_batch(original_texts,return_tensors='pt').to(torch_device)
  translated = model.generate(**tokens)
  translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return translated_texts

def back_translate(texts, language_src, language_dst):
  """Implements back translation"""
  translated = translate(texts, tmp_lang_model, tmp_lang_tokenizer, language_dst)
  back_translated = translate(translated, src_lang_model, src_lang_tokenizer, language_src)
  return back_translated
writer=open('./files/train-fasttxt-dbpedia-AugNMT.train','a')
with open('./files/train-fasttxt-dbpedia5.train','r') as file:
    lines=file.readlines()
    print(len(lines))
    texts=[]
    labels=[]
    for line in lines:
        labels.append(line[0:10])
        texts.append(line[11:-1])
    current=0
    while current<=len(lines):
        print(current)
        if current<=len(lines):
           back_texts = back_translate(texts[current:current+10], "en", "fr")
        else:
           back_texts = back_translate(texts[current:len(lines)-1], "en", "fr")
        for i in range(len(back_texts)):
           writer.write(labels[i]+'\t'+ back_texts[i]+'\n')
        writer.flush()
        current+=10
writer.close()

