from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form.to_dict()

        # Create a DataFrame from the user input
        user_data = pd.DataFrame([user_input])

        # Make predictions using the trained model
        prediction = model.predict(user_data)

        # Display the prediction on the result page
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
