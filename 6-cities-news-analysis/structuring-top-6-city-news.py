import pandas as pd
import plotly.express as px
import numpy as np
import threading
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup
import seaborn as sns
import matplotlib.pyplot as plt
import os
import umap
import matplotlib.pyplot as plt
import glob
files=["city.delhi-corona-news.txt","city.kolkata-corona-news.txt","city.mumbai-corona-news.txt","city.hyderabad-corona-news.txt","city.bengaluru-corona-news.txt","city.chennai-corona-news.txt"]
model=torch.load("emotion_sentiment.pt").to("cuda:0")
labels_mapping={'sadness': 4, 'joy': 2, 'anger': 0, 'fear': 1, 'surprise': 5}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
def preproc(file):
    delhi_news=d=pd.read_csv(file,sep="  ",header=None,names=["publish_date","headline_text"])
    delhi_news.index=pd.to_datetime(pd.Series([str(i)[:4]+"-"+str(i)[4:6]+"-"+str(i)[6:] for i in delhi_news.publish_date.values]))
    weekly_df=pd.DataFrame()
    timeline=[]
    news=[]
    index=[]
    for week in delhi_news.index.week.unique():
        this_weeks_data = delhi_news[delhi_news.index.week == week]
        news.append([this_weeks_data.headline_text.values[i] for i in range(len(this_weeks_data.headline_text.values))])
        timeline.append(str(this_weeks_data.index[0])+"-"+str(this_weeks_data.index[-1]))
        index.append(str(this_weeks_data.index[0]))
    weekly_df["timeline"]=timeline
    weekly_df["news"]=news
    weekly_df.index=index
    return weekly_df
def emotion_classifier(text):
    test_data=tokenizer.encode_plus(
      text,
      max_length=32,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    pred=model(test_data["input_ids"].to("cuda:0"),test_data["attention_mask"].to("cuda:0"))[0].to("cpu").detach().numpy()
    return np.argmax(pred)
def emotion_postproc_p(news_emotions,emotion,weekly_df):
    number_of_emotion_week=[]
    for i in news_emotions:
        emotion_count_this_week=0
        for j in i:
            if j==emotion:
                emotion_count_this_week+=1
        number_of_emotion_week.append((emotion_count_this_week/len(i))*100)
    return number_of_emotion_week
def emotion_postproc(news_emotions,emotion,weekly_df):
    number_of_emotion_week=[]
    for i in news_emotions:
        emotion_count_this_week=0
        for j in i:
            if j==emotion:
                emotion_count_this_week+=1
        number_of_emotion_week.append(emotion_count_this_week)
    return number_of_emotion_week
def main(f):
    f=files[f]
    weekly_df=preproc(f)
    news_emotions=[[emotion_classifier(j) for j in i] for i in weekly_df.news.values]
    weekly_df["Emotions"]=news_emotions
    weekly_df["fear_percentage"]=emotion_postproc_p(news_emotions,1,weekly_df)
    weekly_df["anger_percentage"]=emotion_postproc_p(news_emotions,0,weekly_df)
    weekly_df["sadness_percentage"]=emotion_postproc_p(news_emotions,4,weekly_df)
    weekly_df["joy_percentage"]=emotion_postproc_p(news_emotions,2,weekly_df)
    weekly_df["surprise_percentage"]=emotion_postproc_p(news_emotions,5,weekly_df)
    weekly_df["fear"]=emotion_postproc(news_emotions,1,weekly_df)
    weekly_df["anger"]=emotion_postproc(news_emotions,0,weekly_df)
    weekly_df["sadness"]=emotion_postproc(news_emotions,4,weekly_df)
    weekly_df["joy"]=emotion_postproc(news_emotions,2,weekly_df)
    weekly_df["surprise"]=emotion_postproc(news_emotions,5,weekly_df)
    weekly_df.to_csv(f.split(".")[1]+"-cleaned.csv")

t1=threading.Thread(target=main,args=(0,))
t2=threading.Thread(target=main,args=(1,))
t3=threading.Thread(target=main,args=(2,))
t4=threading.Thread(target=main,args=(3,))
t5=threading.Thread(target=main,args=(4,))
t6=threading.Thread(target=main,args=(5,))
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()