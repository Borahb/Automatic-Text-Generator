# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 19:24:38 2021

@author: Bhaskar
"""

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import time
from stqdm import stqdm

#open
with open('Chars.pkl','rb') as f:
    chars = pickle.load(f)

with open('Xdata.pkl','rb') as f2:
    x_data = pickle.load(f2)

#vocablength
vocab_len = 37

filename = "Model/model_weights_final.hdf5"
md = load_model(filename)



def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    #removing punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)       



num_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_num = dict((c, i) for i, c in enumerate(chars))

st.title('Automatic Text Generation')


nav = st.sidebar.radio("Menu",["Home","Generate text"])



st.progress_bar = st.empty()




if nav == "Home":
    st.image("Images/work.jpg",width = 500,use_column_width = True,caption = 'Work Flow')
    st.write("This is  an automatic text generator which is trained using LSTM architecture. The model is trained on the book named ' The Yellow Wallpaper ' by Charlotte Perkins Gilman. The book is selected from Project Gutenberg, which is a collection of more than 60k ebooks. So at first the model will take an input sentence from the user and then the user have to specify the maximum number of characters to predict, then this information will be fetched to the model and  output will be generated.")
    st.subheader("Python Libraries Used")
    st.markdown("*  Numpy")
    st.markdown("*  Tensorflow")
    st.markdown("*  Nltk")
    st.markdown("*  Pickle")
    st.markdown("*  Streamlit")
    st.markdown("*  Htbuilder")
    
if nav == "Generate text":
    st.header("Generate Text")
    inp = st.text_input("Write a sentence:")
    inp = tokenize_words(inp)
    inp.strip()
    inp = inp[:100]
    inp = [char_to_num[char] for char in inp]
        
    val = st.number_input("Enter the range of characters:",0,1000,step = 100)
    
    
    
    if st.button("Generate"):
        #progress = st.progress(0)
        output_str = ""
        for i in stqdm(range(val)):
            time.sleep(0.5)
            
            x = np.reshape(inp, (1, len(inp), 1))
            x = x / float(vocab_len)
            prediction = md.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = num_to_char[index]
            output_str  += result
                

            inp.append(index)
            inp = inp[1:len(inp)]
        st.write(output_str, end = '')
        st_title = st.empty()
        st_progress_bar = st.empty()


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 80px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="30",
        opacity=1
    )

    style_hr = styles(
        
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)   
    
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def footer():
    myargs = [
        "Made",
        " with ❤️ by ",
        link("https://github.com/Borahb", "@Bhaskar"),
        " & ",
        link("https://github.com/biki321", "@Biki"),
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()