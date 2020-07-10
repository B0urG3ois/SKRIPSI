import numpy as np
import pandas as pd
import pymysql
import pickle
import re
import string
import nltk
import gensim
import xgboost as xgb

# from flask.ext.session import Session
from xgboost import XGBClassifier
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from MeanVectorizer import MeanEmbeddingVectorizer
# from gensim.models.fasttext import FastText
from gensim.models import FastText
from flask.helpers import send_file

app = Flask(__name__)
app.secret_key = "unimedia"
model = pickle.load(open('MODEL_NAME', 'rb')) 
# model_ft = FastText.load_fasttext_format('XGBUpDownSampling800.bin')
# model_ft = FastText.load('XGBUpDownSampling800.bin')
model_ft = FastText.load_fasttext_format('FastText_PreTrained_Model')

connection = pymysql.connect(host='localhost', user='root', password='', database='sentiment')
count = 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/files")
def files():
    return render_template("file.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/sentence', methods=['POST'])
def sentence():
    '''
    For rendering results on HTML GUI
    '''
    
    komen_asli = request.form["Comment"]
    hasil_prediksi = ''

    komentar = komen_asli.lower()
    komentar = re.sub('https?://[A-Za-z0-9./]+', '', komentar)
    komentar = re.sub('http?://[A-Za-z0-9./]+', '', komentar)
    komentar = re.sub(r"\d", "", komentar)
    komentar = re.sub(r'(?<=[,.])(?=[^\s])', r' ', komentar)
    komentar = komentar.translate(str.maketrans('', '', string.punctuation))
    komentar = nltk.tokenize.word_tokenize(komentar)
    komentar = [komentar]

    mean_vec_tr = MeanEmbeddingVectorizer(model_ft)
    trf_vector = mean_vec_tr.transform(komentar)
    vector = xgb.DMatrix(trf_vector)
    prediksi = model.predict(vector)

    if(prediksi==2):
        hasil_prediksi = 'Positive'
    elif(prediksi==1):
        hasil_prediksi = 'Neutral'
    else:
        hasil_prediksi = 'Negative'
    
    return render_template('predict.html', comment_content="{}".format(komen_asli), prediction_text="{}".format(hasil_prediksi))

@app.route('/show_text', methods=['GET', 'POST'])
def show_text():
    file = request.files['inputFile']
    df_file = pd.read_excel(file)
    cursor = connection.cursor()

    sql = "DROP TABLE IF EXISTS `data`"
    cursor.execute(sql)

    cols = "`,`".join([str(i) for i in df_file.columns.tolist()])

    sql = "CREATE TABLE data ("+cols+" TEXT);"
    cursor.execute(sql)

    for i,row in df_file.iterrows():
        sql = "INSERT INTO `data` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))
        connection.commit()
    
    return render_template(
        "file.html",
        total=len(df_file),
        tables=[df_file.to_html(classes="table table-striped", header="true", index=False)],
        titles=df_file.columns.values,
        shape=count
    )

@app.route("/predicted", methods=['GET', 'POST'])
def predicted():
    cursor2 = connection.cursor()
    sql = "SELECT * from `data`"
    cursor2.execute(sql)
    result = cursor2.fetchall()
    texts = []
    pos = 0
    net = 0
    neg = 0

    for i in result:
        texts.append(i)

    #pre-process
    dfprint = pd.DataFrame()
    dfprint['Text'] = texts

    komentar = dfprint['Text'].astype(str)
    komentar = komentar.apply(lambda x: x.lower())
    komentar = komentar.apply(lambda x: re.sub('https?://[A-Za-z0-9./]+','',x))
    komentar = komentar.apply(lambda x: re.sub('http?://[A-Za-z0-9./]+','',x))
    komentar = komentar.apply(lambda x: re.sub(r"\d", "", x))
    komentar = komentar.apply(lambda x: re.sub(r'(?<=[,.])(?=[^\s])', r' ', x))
    komentar = komentar.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    komentar = komentar.apply(lambda x: re.sub(r'([^\s\w]|_)+', '', x))
    komentar = komentar.apply(lambda x: nltk.tokenize.word_tokenize(x))

    dfprint['tokenized'] = komentar
    dfprint = dfprint[dfprint.tokenized.astype(bool)]
    dfprint = dfprint.dropna()

    #prediction process
    mean_vec_tr = MeanEmbeddingVectorizer(model_ft)
    trf_vector = mean_vec_tr.transform(dfprint['tokenized'])
    vector = xgb.DMatrix(trf_vector)
    prediksi = model.predict(vector)

    # mean_vec_tr = MeanEmbeddingVectorizer(model_ft)
    # hasil_vector = mean_vec_tr.transform(dfprint['tokenized'])
    # predictions = model.predict(hasil_vector)

    dfprint['Sentiment'] = prediksi
    dfprint['Sentiment'] = dfprint['Sentiment'].replace([2, 1, 0], ['Positive', 'Neutral', 'Negative'])
    dfpred = dfprint[['Text', 'Sentiment']]

    df_export = dfpred

    output_file = 'predict_data.xlsx'
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    df_export.to_excel(writer, sheet_name="Prediction")
    workbook = writer.book
    worksheet1 = writer.sheets['Prediction']
    writer.save()
    session['opt_file'] = output_file

    net_count = len(dfprint[dfprint['Sentiment'] == 'Neutral'].index)
    neg_count = len(dfprint[dfprint['Sentiment'] == 'Negative'].index)
    pos_count = len(dfprint[dfprint['Sentiment'] == 'Positive'].index)

    totalData = len(prediksi)
    for j in prediksi:
        if j == 1:
            pos = pos + 1
        elif j == 0:
            net += 1
    neg = totalData - pos - net

    posPercentage = (pos/totalData)*100
    negPercentage = (neg/totalData)*100
    netPrecentage = (net/totalData)*100
 
    donePredict = True

    return render_template(
        # "predicted.html",
        "file.html",
        net_count = net_count,
        neg_count = neg_count,
        pos_count = pos_count,
        posP = posPercentage,
        negP = negPercentage,
        netP = netPrecentage,
        tables2=[dfpred.to_html(classes="table table-striped", header="true", index=False)],
        titles=dfpred.columns.values
    )

@app.route("/download", methods=['GET', 'POST'])
def download():
    output_file = session.get('otpt_file', None)
    return send_file('predict_data.xlsx', attachment_filename='output.xlsx', as_attachment=True,mimetype='text/xlsx')

if __name__ == "__main__":
    app.run(debug=False)