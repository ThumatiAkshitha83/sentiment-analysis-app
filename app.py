from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Transform input
    transformed = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(transformed)[0]
    
    return render_template('index.html', prediction_text=f"Sentiment: {prediction}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

