from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your model
with open("heart_failure_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["POST", "GET"])
def index():
    result = ""
    error = ""
    if request.method == "POST":
        try:
            # Fetch and validate inputs
            age = float(request.form["age"])
            ejection_fraction = float(request.form["ejection_fraction"])
            serum_creatinine = float(request.form["serum_creatinine"])
            serum_sodium = float(request.form["serum_sodium"])

            # Check for valid input ranges (customize as needed based on your data)
            if not (0 <= age <= 120):
                error = "Please enter a valid age between 0 and 120."
            elif not (10 <= ejection_fraction <= 80):
                error = "Ejection fraction should be between 10% and 80%."
            elif not (0.5 <= serum_creatinine <= 5.0):
                error = "Serum creatinine should be between 0.5 and 5.0 mg/dL."
            elif not (120 <= serum_sodium <= 150):
                error = "Serum sodium should be between 120 and 150 mEq/L."
            else:
                # Valid inputs - make prediction
                features = np.array([[age, ejection_fraction, serum_creatinine, serum_sodium]])
                prediction = model.predict(features)

                # Interpret prediction
                if prediction[0] == 1:
                    result = "High risk of heart failure."
                else:
                    result = "Low risk of heart failure."

        except ValueError:
            error = "Please enter valid numeric input for all fields."

    return render_template("index.html", result=result, error=error)
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
