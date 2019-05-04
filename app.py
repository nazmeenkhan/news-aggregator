from flask import Flask, render_template, jsonify, request,url_for
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np 
import re
import sklearn
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix,precision_score
global vectorize,nb

app = Flask(__name__)

@app.route("/")
def index():
    print("rendering template")
    return render_template('index.html')

@app.route("/classify", methods=["POST"])
def classify():

    try: 
        print("classification")
        query = [request.form['news']]
        print(type(query))
        print(query)
        Y_predict1=nb.predict(vectorize.transform(query))
        print(Y_predict1)
        if(Y_predict1=="e"):
            Y_predict1="Entertainment"
        else:
            Y_predict1="Business"
        return jsonify({'classification': str(Y_predict1)})

    except:

    	return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    nb = joblib.load("101.pkl") # Load "model.pkl"
    vectorize=joblib.load("100.pkl") # Load "vectorizer.pkl"
    print ('Model loaded')    
    app.run(port=port, debug=True)
