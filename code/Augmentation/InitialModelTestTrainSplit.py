# -----------------------------------------------------------
#Vec2Node: Self-training with Tensor Augmentation for Text Classification with Few Labels
# Authors: Sara Abdali, Subhabrata Mukherjee, Evangelos Papalexakis
# (C) ECML-PKDD 2022
# sabda005@ucr.edu
# -----------------------------------------------------------
import os
import csv
import pandas as pd
import random
import math
import shutil
import numpy as np
import argparse
import fasttext
from sklearn.model_selection import train_test_split
from numpy.random import RandomState


def createModel(percentage,task):

    df = pd.read_csv('./data/NLP-dataset/'+task+'/train_clean.csv',sep=',')
    rng = RandomState()
    if percentage=='100':
        text_train=df.iloc[:, 0]
        label_train=df.iloc[:, 1]
    else:
        text_train,_,label_train,_= train_test_split(df.iloc[:, 0],df.iloc[:, 1],train_size=float(percentage)/100.0,random_state=rng)
    print(label_train)
    if os.path.exists('./files/train-fasttxt-'+task+'10.train'):
        os.remove('./files/train-fasttxt-'+task+'10.train')
    with open('./files/train-fasttxt-'+task+'10.train','w') as f1:
        for text,label in zip(text_train,label_train):
            f1.write('__label__'+str(label)+'\t')
            f1.write(text+'\n')
        f1.close()
    test = pd.read_csv('./data/NLP-dataset/'+task+'/test_clean.csv',sep=',')
    if os.path.exists('./files/test-fasttxt-'+task+'.valid'):   
        os.remove('./files/test-fasttxt-'+task+'.valid')      
    with open('./files/test-fasttxt-'+task+'.valid','w') as f1:
        for i in range(len(test)): 
            f1.write('__label__'+str(test.iloc[i,1])+'\t')    
            f1.write(test.iloc[i,0]+'\n')
        f1.close()
    model = fasttext.train_supervised('./files/train-fasttxt-'+task+'10.train') 
    model.save_model('./model/fasttext_'+task+'_init_10.bin')

if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("--percentage", "-p", help="Percentage of train set between 1-100")
   parser.add_argument("--task", "-t", help="Name of dataset agnews, imdb, sst2 and dbpedia")
   args = parser.parse_args()
   createModel(**vars(args))
