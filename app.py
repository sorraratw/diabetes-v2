import pickle
import numpy as np

from flask import Flask, render_template, request
from keras.models import load_model


app = Flask(__name__)

classifier = load_model('diabetesModel.h5')
sc_x = pickle.load(open('scaler.pkl', 'rb'))





@app.route('/')
def main():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def getPredict():
    x1 = request.form['x1']
    x2 = request.form['x2']
    x3 = request.form['x3']
    x4 = request.form['x4']
    x5 = request.form['x5']
    x6 = request.form['x6']
    x7 = request.form['x7']
    x8 = request.form['x8']
    x9 = request.form['x9']
    x10 = request.form['x10']
    x11 = request.form['x11']
    x12 = request.form['x12']
    x13 = request.form['x13']
    x14 = request.form['x14']
    x15 = request.form['x15']
    x16 = request.form['x16']




    XTest = np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 ]], dtype=np.float64)
    m_test = sc_x.transform(XTest)
    #predicted = classifier.predict(m_test)[0]
    predicted = classifier.predict(m_test)
    predicttext = predicted[0][0]
    return render_template('index.html',
                           prediction_text=f'Predicted: {predicttext * 100:.2f}%')


if __name__ == '__main__':
    app.run(debug=True)