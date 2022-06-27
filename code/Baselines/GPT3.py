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

If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.
WARNING:tensorflow:Model was constructed with shape (None, 80) for input Tensor("input_ids:0", shape=(None, 80), dtype=int32), but it was called on an input with incomp        counter=0
        for line in lines:
            labels.append(line[0:10])
            texts.append(line[11:-1])
        for i in range(len(lines)):
            print(i)
            test_tokenized = tokenizer.encode_plus(texts[i], return_tensors="pt")
            test_input_ids  = test_tokenized["input_ids"]
            test_attention_mask = test_tokenized["attention_mask"]
            t5_model.eval()
            beam_outputs = t5_model.generate(
                          input_ids=test_input_ids,attention_mask=test_attention_mask,
                          max_length=512,
                          early_stopping=True,
                          num_beams=4,
                          num_return_sequences=4,
                          no_repeat_ngram_size=2
                          )

            for beam_output in beam_outputs:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                pre_label=classifier.predict(sent)[0]
                if pre_label[0]==labels[i]:
                      writer.write(labels[i]+'\t'+ sent+'\n')
                      prevSents.add(sent)
                      counter+=1
            if counter>=int(iteration):
                Retrain(task,prevSents)
                classifier = fasttext.load_model('./data/fasttext_'+task+'.bin')
                counter=0
            writer.flush()


if __name__ == "__main__":

   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument("--percentage", "-pl", help="set test size between 1-100")
   parser.add_argument("--iteration", "-i", help="number of iterations for self-training")
   parser.add_argument("--task", "-t", help="name of dataset imdb, agnews, sst2 and dbpedia")
   args = parser.parse_args()
   logging.info("Experiment information:  Retrain the model after adding "+str(args.percentage)+" new sentences\n"+
                                         "Retrain the model after adding "+str(args.percentage)+" new sentences\n"+
                                         "Experiment on "+str(args.task)+" dataset\n")
   GenerateSent(**vars(args))

