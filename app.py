# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:32:53 2022

@author: AKammari
"""

import numpy as np
from flask import Flask, request,render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    #output = prediction[0]
 
    return render_template('index2.html', prediction_text='The species is  {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug = True)
# -*- coding: utf-8 -*-

