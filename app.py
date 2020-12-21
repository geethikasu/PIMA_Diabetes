from flask import Flask,render_template,request
import pickle
import numpy as np

file ="diabetes_model.pkl"
model = pickle.load(open(file,'rb'))

app =Flask(__name__)

@app.route('/')
def home():
      return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        preg = int(request.form['Pregnancies'])
        glu = int(request.form['Glucose'])
        bp = int(request.form['BloodPressure'])
        skthick = int(request.form['SkinThickness'])
        insu = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        input = np.array([[preg,glu,bp,skthick,insu,bmi,dpf,age]])
        output = model.predict(input)

        return render_template('result.html',pred=output)

if __name__ == '__main__':
	app.run(debug=True)



