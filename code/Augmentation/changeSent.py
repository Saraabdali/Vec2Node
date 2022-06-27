# -----------------------------------------------------------
#Vec2Node: Self-training with Tensor Augmentation for Text Classification with Few Labels
# Authors: Sara Abdali, Subhabrata Mukherjee, Evangelos Papalexakis
# (C) ECML-PKDD 2022
# sabda005@ucr.edu
# -----------------------------------------------------------
import pandas as pd
import h5py
import csv
import numpy as np
import scipy.io as spio
import fasttext
import random
import os
import string
import logging
import argparse
import nltk
import math
import tfidf
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from difflib import SequenceMatcher 

def Retrain(task,prevSents,embedding):
    concat = ""
    files=['./files/train-fasttxt-'+task+'5.train','./files/train-fasttxt-'+task+'_embed_'+embedding+'_Aug.train']
    for file in files:
        concat += open(file).read()
    with open('temp.train','w') as tf:
        tf.write(concat)        
    classifier=fasttext.train_supervised('temp.train')
    classifier.save_model('./data/fasttext_'+task+'.bin')
    

def Similarity_check(newSent,sent,prevSents):
    repetition_check=False
    if(SequenceMatcher(None, newSent,sent).ratio()<0.99):
        for prev in prevSents:
            if( SequenceMatcher(None, newSent,prev).ratio()>0.99):
                repetition_check=True
                return True
    if (repetition_check==False):
        return False       
##############################################################################################################################
def GenerateDict(task):
    Dict={}
    filename='./data/NLP-dataset/'+task+'/unsup.txt'
    counter=0
    with open(filename,'r') as f:
        for line in f:
            words = line.replace('.','').replace('\'','').replace(',','').lower().split()
            for word in words:
                if word not in Dict:
                    Dict[word] = counter
                    counter+=1
    return Dict,counter
def GenerateSent(percentage,iteration,task,neighbors,embedding):
    count=0
    concept_drifts=0
    Stopwords=stopwords.words('english')
    filename='./files/train-fasttxt-'+task+'5.train' #opening default  train file
    df1=pd.read_csv(filename,sep='\t')
    lbls=df1.iloc[:, 0]
    sents=df1.iloc[:, 1]
    print(len(sents))
    direction=['lr','rl']#
  
    if embedding=='fasttext':
        model = fasttext.train_unsupervised('./data/NLP-dataset/'+task+'/unsup.txt',minn=2, maxn=5, dim=50)
    elif embedding=='glove':
       from glove import Corpus, Glove
       from gensim.models.word2vec import LineSentence
       # creating a corpus object
       corpus = Corpus() 
       #training the corpus to generate the co occurence matrix which is used in GloVe
       corpus.fit(LineSentence('./data/NLP-dataset/'+task+'/unsup.txt'), window=3)
       glove = Glove(no_components=5, learning_rate=0.05)
       glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
       glove.add_dictionary(corpus.dictionary)
    
    elif embedding=='word2vec':
        from gensim.models.word2vec import LineSentence
        from gensim.models import Word2Vec

        model = Word2Vec(min_count=2, window=3,vector_size =50,workers=10)

        model.build_vocab(LineSentence('./data/NLP-dataset/'+task+'/unsup.txt'), progress_per=10000)
        model.train(LineSentence('./data/NLP-dataset/'+task+'/unsup.txt'), total_examples=model.corpus_count, epochs=10, report_delay=1)
    elif embedding=='tfidf':
        with open('tf_idf_dict.pickle', 'rb') as handle:
             Dict= pickle.load(handle)
        with open('tf_idf_mat.pickle', 'rb') as handle:
             tfidf_mat= pickle.load(handle)
        [U,S,V]=svd(tfidf_mat)
        factorV=V.transpose()
    elif embedding=='random':                     
        Dict,Len=GenerateDict(task)
        random_indices=[]
        for i in range(0,int(neighbors)):
            random_indices.append(random.randint(0,Len-1))
            Dict_keys=list(Dict.keys())
            Dict_values=list(Dict.values())

    elif embedding=='tensor'or embedding=='random':
        data=spio.loadmat('./dictionary/KNN_'+task+'_CP.mat')#loading KNN graph generated with CP/PARAFAC embedding
        factorMatA=data['A']#word embedding Ai
        df=pd.read_csv('./dictionary/dictionary_'+task+'.tsv',usecols=['key','value'],sep='\t')
        Dict=dict(zip(list(df['key']),list(df['value'])))
    elif embedding=='hypergraph_tensor':
        df2=pd.read_csv('./Vec2Node/data/NLP-dataset/dbpedia/all_dbpedia.csv',sep=',')
        data=spio.loadmat('./dictionary/KNN_'+task+'_CP.mat')#loading KNN graph generated with CP/PARAFAC embedding
        factorMatA=data['A']#word embedding Ai
        factorMatB=data['B']
        factorMatC=data['C']
        df=pd.read_csv('./dictionary/dictionary_'+task+'.tsv',usecols=['key','value'],sep='\t')
        Dict=dict(zip(list(df['key']),list(df['value'])))
    fn='./model/fasttext_'+task+'_init.bin'
    classifier = fasttext.load_model(fn)#loading model trained on initial train set
    if os.path.exists('./files/train-fasttxt-'+task+'_embed_'+embedding+'_Aug.train'):
        os.remove('./files/train-fasttxt-'+task+'_embed_'+embedding+'_Aug.train')
    with open('./files/train-fasttxt-'+task+'_embed_'+embedding+'_Aug.train','w') as f1:
        perc=[1,2,3,4]
        prevSents=set()    
        for i in range(len(sents)):
          print(i)
          for pl in perc:
            #prevSents=set()
            words=word_tokenize(sents[i])
            words=[word.lower() for word in words]
            stop=math.ceil(len(words)/pl)
            for Dir in direction:
                newSent=['']*int(neighbors)
                if Dir=='rl':
                    words.reverse()
                for j in range(len(words)):
                    if j<stop:
                        if embedding=='glove':
                            from scipy import spatial
                                      
                            if words[j] in  glove.dictionary.keys():
                               nearest_words=sorted(glove.dictionary.keys(), key=lambda word: spatial.distance.euclidean(glove.word_vectors[glove.dictionary[word]]
                                                                                              ,glove.word_vectors[glove.dictionary[words[j]]]))[1:int(neighbors)+1] 
                            else:
                               nearest_words=[words[j] for k in range(int(neighbors))]          
                        elif embedding=='word2vec':

                            if words[j] in  list(model.wv.index_to_key):
                               nearest_words=model.wv.most_similar(positive=words[j], topn=int(neighbors))
                               nearest_words=[a_tuple[0] for a_tuple in nearest_words]
                            else:
                               nearest_words=[words[j] for k in range(int(neighbors))]             
                           
                        elif embedding=='fasttext':
                            nearest_words=[a_tuple[1] for a_tuple in  model.get_nearest_neighbors(words[j])][0:int(neighbors)]
    
                        elif embedding=='random':
                            nearest_words=[Dict_keys[Dict_values.index(i)] for i in random_indices]

                        elif embedding=='hypergraph_tensor':
                            snn=2
                            nearest_sents=factorMatB[i,:]
                            ns=np.nonzero(nearest_sents)[1]
                            words_nearest_sents=[]
                            dist=[]
                            for m in range(snn):
                                new_sent=df.iloc[ns[m],0]
                                tokens=[word for word in word_tokenize(new_sent) if word in Dict.keys()]
                                words_nearest_sents.extend(tokens)
                            if words[j] not in Dict.keys():
                                nearest_words=[words[j] for k in range(int(neighbors))]
                            else:
                                for w in range(len(words_nearest_sents)):
                                    value1=Dict[words[j]]
                                    value2=Dict[words_nearest_sents[w]]
                                    dist.append(np.linalg.norm(factorMatC[value1]-factorMatC[value2]))
                                nearest_words_idx=sorted(range(len(dist)), key = lambda sub: dist[sub])[-int(neighbors):]
                                nearest_words_idx=[ele for ele in reversed(nearest_words_idx)]   
                                nearest_words=[words_nearest_sents[nearest_words_idx[l]] for l in range(int(neighbors))]       
                                
                        elif embedding=='tensor':
                            if words[j] in list(df['key']):
                                value=Dict[words[j]]
                                nearest_words=factorMatA[value,:]
                                nw=np.nonzero(nearest_words)[1]
                                nearest_words=[df.iloc[nw[k]]['key'] for k in range(int(neighbors))]
                            else:
                                nearest_words=[words[j] for k in range(int(neighbors))]

                        elif embedding=='tfidf':
                            if words[j] in list(df['key']):
                                value=Dict[words[j]]
                                nearest_words=factorV[value,:]
                                nw=np.nonzero(nearest_words)[1]
                                nearest_words=[df.iloc[nw[k]]['key'] for k in range(int(neighbors))]
                            else:
                                nearest_words=[words[j] for k in range(int(neighbors))]
                        newSent=[(word,word+' '+sent) for word,sent in zip(nearest_words,newSent)]
                        newSent=[element[1] for element in newSent]#newSent=list(zip(*newSent))          
                        
                    else:
                        newSent=[words[j]+' '+sent for sent in newSent]#keeping rest of the sentesnce intact   
        
                for k in range(len(newSent)):
                    if Dir=='lr':
                        newSent[k]=' '.join(reversed(newSent[k].split(' '))).strip()
                    #print("new: "+newSent[k])
                    #print("original: "+sents[i])

                    pre_label=classifier.predict(newSent[k])[0]       
                    if pre_label[0]==lbls[i] and newSent[k] not in prevSents:# Similarity_check(newSent[k],sents[i],prevSents)==False: #concept drift checking
                         count+=1
                         with open('./files/train-fasttxt-'+task+'_embed_'+embedding+'_Aug.train','a') as f1:
                             f1.write(pre_label[0]+'\t'+newSent[k]+'\n')
                         f1.close()
                         prevSents.add(newSent[k])
                         if count%int(iteration)==0:
                             logging.info('Retraining in process...')
                             Retrain(task,prevSents,embedding)
                             classifier = fasttext.load_model('./data/fasttext_'+task+'.bin')
                             logging.info('Retraining finished!')
                    else:
                        concept_drifts+=1 
        logging.info("Number of concept drift: "+str(concept_drifts))
#########################################################################################################################         
if __name__ == "__main__":

   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument("--percentage", "-pl", help="set test size between 1-100")
   parser.add_argument("--iteration", "-i", help="number of iterations for self-training")
   parser.add_argument("--task", "-t", help="name of dataset imdb, agnews, sst2 and dbpedia")
   parser.add_argument("--neighbors", "-n", help="number of neighbors in KNN graph to use for replacement")
   parser.add_argument("--embedding", "-e", help="embedding tensor, word2vec, fasttext, glove and random")
   args = parser.parse_args()
   logging.info("Experiment information:  Percentage of change is "+str(args.percentage)+'%\n'+ 
                                         "Retrain the model after adding "+str(args.percentage)+" new sentences\n"+
                                         "Experiment on "+str(args.task)+" dataset\n"+
                                         "Using "+str(args.neighbors)+" most similar neighbors\n"+
                                         "Based on "+str(args.embedding)+" embedding space\n")
   GenerateSent(**vars(args))

