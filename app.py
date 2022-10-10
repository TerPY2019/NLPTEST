from lib2to3.pgen2 import token
from flask import Flask,render_template,request,Markup
from matplotlib.pyplot import text
from textblob import TextBlob
import os
import test
import spacy
from  transformers.file_utils import is_tf_available,is_torch_available, is_torch_tpu_available
from  transformers import Trainer, TrainingArguments
import numpy as np
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
from spacy import displacy
app = Flask(__name__)
@app.route('/')
def hello():
    return render_template('homepace.html')

paths = os.getcwd()+"\savefile"
@app.route('/conver_nlp',methods=["post"])
def method_name():
    namefile = request.files.getlist("textfire[]")
    stext = request.form.get('stext')
    tatalfile = []
    for i in namefile:
        joinpath = os.path.join(paths,i.filename)
        i.save(joinpath)
        tatalfile.append(joinpath)
    textnlp = test.bow(tatalfile)
    addtoken = textnlp.ctoken()
    strtext = textnlp.dicttext(stext)
    topbow = textnlp.topfivebow()
    toptfidf = textnlp.topfivetfidf()
    return render_template('homepace.html',show=addtoken,setext=strtext,bowtopfive=topbow,tfidftopfive=toptfidf)

@app.route('/spacytext',methods=["post","get"])
def spacycode():
    nametext = request.form.get('spatext')
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(nametext)
    showNER = displacy.render(doc,style="ent")
    return render_template('homepace.html',nershow=Markup(showNER))

@app.route('/testmodel',methods=["post","get"])
def facknew():
    nametext1 = request.form.get('spatext1')
    facknewtest  = get_prediction(nametext1, convert_to_label=True)
    return render_template('homepace.html',facknewtest1=facknewtest)
def get_prediction(text, convert_to_label=False):
    model_path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/fake-news-dataset-ter-model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512,return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())
@app.route('/sentimentanalysis',methods=["post","get"])
def sentimentanalysisfu():
    nametext2 = request.form.get('spatext2')
    chacksentiment = chack(nametext2)
    return render_template('homepace.html',chacksentiment=chacksentiment)
def chack(nametext):
     #สร้างtext object
    blob_two_cities = TextBlob(nametext)
    if blob_two_cities.sentiment[0] < 0:
        text1 = "Negative"
    elif blob_two_cities.sentiment[0] == 0:
        text1 = "Neutral"
    else:
        text1 = "positive"
    return(text1)
if __name__ == '__main__':
    app.run(debug=True,host="192.168.56.1",port="80")