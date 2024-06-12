from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = 'Good Quality' if prediction == 1 else 'Bad Quality'
    return render_template('result.html', prediction_text=f'The wine quality is predicted to be: {output}')

if __name__ == "__main__":
    app.run(debug=True)