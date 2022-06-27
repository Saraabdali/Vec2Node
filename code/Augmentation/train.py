# -----------------------------------------------------------
#Vec2Node: Self-training with Tensor Augmentation for Text Classification with Few Labels
# Authors: Sara Abdali, Subhabrata Mukherjee, Evangelos Papalexakis
# (C) ECML-PKDD 2022
# sabda005@ucr.edu
# -----------------------------------------------------------

import os
import fasttext
import argparse
from sklearn.metrics import accuracy_score
from statistics import mean
from statistics import stdev


def classify(iterations,task,Aug):
   acc=[]
   for i in range(int(iterations)):
     if Aug=="no":
        model = fasttext.train_supervised('./files/train-fasttxt-'+task+'.train',dim=150)
     else:
        model = fasttext.train_supervised('./test.train' ,dim=50)
     with open('./files/test-fasttxt-'+task+'.valid', 'r') as test:
        line=test.readline()
        count=0.0
        num_sample=0.0
        while line:
             label=line[0:10]
             text=line[11:-1]
             result=model.predict(text)
             if result[0][0]==label:
                count=count+1
             num_sample=num_sample+1
             line=test.readline()
        test.close()
        acc.append(count/num_sample)
   print ("Accuracy= "+str(mean(sorted(acc, reverse=True)[0:i]))+"("+str(stdev(sorted(acc, reverse=True)[0:i]))+")")
   print ("Top1 Accuracy= "+str(sorted(acc, reverse=True)[0]))


if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("--iterations", "-i", help="number of iterations")
   parser.add_argument("--task", "-t", help="Dataset could be imdb, agnews or sst2")
   parser.add_argument("--Aug", "-a", help="Aug could be yes or no")
   args = parser.parse_args()
   classify(**vars(args))                    
