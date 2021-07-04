# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:40:59 2021

@author: Bhaskar
"""



import sys
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

#open
with open('Chars.pkl','rb') as f:
    chars = pickle.load(f)

with open('Xdata.pkl','rb') as f2:
    x_data = pickle.load(f2)

#vocablength
vocab_len = 37


num_to_char = dict((i, c) for i, c in enumerate(chars))

char_to_num = dict((c, i) for i, c in enumerate(chars))  


 

#print(pattern)

#loading the model
filename = "model_weights_final.hdf5"
md = load_model(filename)


    

def Predict(pattern,ran):
    for i in range(ran):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_len)
        prediction = md.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = num_to_char[index]


        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        
        
        
        print(result,end = '')
        
        
def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    #removing punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)


inp = input('Write a seentence:')
inp = tokenize_words(inp) 
#print(inp)  
inp.strip()
inp = inp[:100]
inp = [char_to_num[char] for char in inp]
#len(inp) 

Predict(inp,100)






