from flask import Flask,render_template,request
import pandas as pd

import pickle
import numpy as np
data=pd.read_csv('Cleaned_data.csv')
pipe=pickle.load(open("ri.pkl",'rb'))
app=Flask(__name__,template_folder='templates')
@app.route('/')
def main():
    locations=sorted(data['location'].unique())

    return render_template('index.html',locations=locations)
@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')

    bath = request.form.get('bath')

    sqft = request.form.get('total_sqft')
    print(location,bhk,bath,sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0] * 1e5
    return str(np.round(prediction,2))
if __name__=="__main__":
    app.debug=True
    app.run()
