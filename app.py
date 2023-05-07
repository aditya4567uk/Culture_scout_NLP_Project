# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@aditya
"""
import re
import sys
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import docx2txt
import spacy
from spacy import displacy
from collections import Counter
from tqdm.autonotebook import tqdm
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
#from multi_rake import Rake

st.set_page_config(page_title="CultureScout: NLP-Driven Enterprise Culture Analytics Hackathon by TEAM-->RaAd",page_icon=":tada:",layout="wide")

def lottie_req(url):
    r= requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

obtained_txt = docx2txt.process(r'testNlp.docx')
doc_uploader = obtained_txt


email = re.compile(r'[a-zA-Z0-9-\.]+@[a-zA-Z-\.]*\.(com|ac|edu|net)')
phone_no = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')

emails = email.finditer(doc_uploader)
phone_nos = phone_no.finditer(doc_uploader)

email_list =[]
def return_emails(emails):
    for gotEmails in emails:
        email_list.append(gotEmails.group(0))

phone_list=[]
def return_phone(phone_nos):
    for gotPhoneNums in phone_nos:
        phone_list.append(gotPhoneNums.group(0))


nlp = spacy.load("en_core_web_lg")
nlp_sm = spacy.load("en_core_web_sm")
doc = nlp(doc_uploader)
just_text = doc_uploader
docs = list(tqdm(nlp.pipe(just_text), total=len(just_text)))

#[(i.text, i.ent_iob_ + "-" + i.ent_type_) for i in doc[0:30]]

all_persons = []
def return_person(doc):
    for d in doc:
        persons = [ent.text for ent in doc if ent.ent_type_ == "PERSON"]
        all_persons.extend(persons)
    Counter(all_persons).most_common(15)


all_orgs = []
def return_orgs(doc):
    for d in doc:
        orgs = [ent.text for ent in doc if ent.ent_type_ == "ORG"]
        all_orgs.extend(orgs)

    Counter(all_orgs).most_common(15)


specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
result_sentiment = []

#custom_data=['i was about to complete my degree and suddenly my dog woke me up']

def return_sentiment(inp1):
    global data
    data = specific_model(inp1)
    #result_sentiment=data
    #return data

obtained_txt2 = "Data mining, also known as knowledge discovery in data (KDD), is the process of uncovering patterns and other valuable information from large data sets. Given the evolution of data warehousing technology and the growth of big data, adoption of data mining techniques has rapidly accelerated over the last couple of decades, assisting companies by transforming their raw data into useful knowledge. However, despite the fact that that technology continuously evolves to handle data at a large-scale, leaders still face challenges with scalability and automation.Data mining has improved organizational decision-making through insightful data analyses. The data mining techniques that underpin these analyses can be divided into two main purposes; they can either describe the target dataset or they can predict outcomes through the use of machine learning algorithms. These methods are used to organize and filter data, surfacing the most interesting information, from fraud detection to user behaviors, bottlenecks, and even security breaches.When combined with data analytics and visualization tools, like Apache Spark, delving into the world of data mining has never been easier and extracting relevant insights has never been faster. Advances within artificial intelligence only continue to expedite adoption across industries."
#rake = Rake()



def return_keywords(obtained_txt2):
    global keywords
    #keywords = rake.apply(obtained_txt2)

lottie_url = lottie_req("https://assets1.lottiefiles.com/packages/lf20_2glqweqs.json")


st.subheader("Hi 🤞 We are TEAM-->RaAD -Aditya Kumar & Rahul Patil")
st.title("CultureScout: NLP-Driven Enterprise Culture Analytics Hackathon by TEAM-->RaAd")
#st.write("Functionality of this web page --> EXTRACTING INFO LIKE Name , organizations, Email, Phone Numbers, from document\n Sentimental Classification and Visual presentation\n Context and Content identification for a given corpus\n DeepLearnign based Question and answering feature from the given long paragraph of text")

with st.container():
    st.write("------")
    left_column,right_column = st.columns(2) #2 is size of column we want to give
    with left_column:
        st.header("What this app can do")
        st.write(
            """
            
            -EXTRACTING INFO LIKE Name , organizations, Email, Phone Numbers
            -Sentimental Classification and Visual presentation
            -Context and Content identification for a given corpus
            -DeepLearning based Question and answering feature from the given long paragraph of text
            
            """
            
            )
    
    with right_column:
        st_lottie(lottie_url,height=400,key="coding")

with st.container():
    st.write("------")
    st.subheader("Uploading files")
    left_column1,right_column1 = st.columns(2)
    
    with left_column1:
        doc_uploader = st.file_uploader("Please upload a doc file",type=["docx"])
        btn1 = st.button("Extract Emails",on_click=return_emails(emails))
        btn2 = st.button("Extract PhoneNo",on_click=return_phone(phone_nos))
        btn3 = st.button("Extract persons",on_click=return_person(doc))
        btn4 = st.button("Extract Organizations", on_click=return_orgs(doc))
        st.subheader("Taking user input for sentiment analysis")
        inp1 = st.text_input('enter text for sentiment analysis')
        btn5 = st.button("Sentiment Analysis",on_click=return_sentiment(inp1))
        inp2 = st.text_input("Enter text for keyword identification")
        btn6 = st.button("enter pararapgh",on_click=return_keywords(inp2))
    
    with right_column1:
        st.subheader("OUTPUT")
        if btn1:
            st.write(email_list)
        if btn2:
            st.write(phone_list)
        if btn3:
            st.write(Counter(all_persons).most_common(15))
        if btn4:
            st.write(Counter(all_orgs).most_common(15))
        if btn5:
            st.write(data)
        if btn6:
            st.write(keywords)
        