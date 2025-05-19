from flask import Flask, request, render_template
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])  # Transform text using vectorizer
    prediction = model.predict(data)[0]     # 0 or 1
    
    # Convert numeric prediction to text
    label = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template('result.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)

