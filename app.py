import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder

# create flask app
app = Flask(__name__)

# Load the pickle model
classification = pickle.load(open("classification.pkl", "rb"))
regression = pickle.load(open("regression.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("home.html")


@app.route('/discount')
def discount():
    return render_template("classification.html")


@app.route('/sales_target')
def sales_target():
    return render_template("regression.html")


@app.route("/predict_discount", methods=["POST"])
def predict_discount():
    val_list = []
    for y in request.form.values():
        y_list = [y]
        le = LabelEncoder()
        y_new = le.fit_transform(y_list)
        val_list.append(float(y_new))
    features = [np.array(val_list)]
    prediction = classification.predict(features)
    return render_template("classification.html", prediction_text="Discount is {}".format(prediction))


@app.route("/predict_sales_target", methods=["POST"])
def predict_sales_target():
    val_list = []
    for x in request.form.values():
        if x == 'Consumer':
            val_list.append(float(0))
        elif x == 'Corporate':
            val_list.append(float(1))
        elif x == 'Home Office':
            val_list.append(float(2))
        else:
            val_list.append(float(x))

    features = [np.array(val_list)]
    prediction = regression.predict(features)
    return render_template("regression.html", prediction_text="Target Sales is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
