from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from wtforms import Form, FloatField, validators

# Load mô hình Naive Bayes đã được lưu
model = joblib.load('model/naive_bayes_model.pkl')

# Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')



# Form để nhập các đặc tính của hoa
class IrisForm(Form):
    sepal_length = FloatField('Sepal Length (cm)', [validators.InputRequired()])
    sepal_width = FloatField('Sepal Width (cm)', [validators.InputRequired()])
    petal_length = FloatField('Petal Length (cm)', [validators.InputRequired()])
    petal_width = FloatField('Petal Width (cm)', [validators.InputRequired()])


# Trang chủ với form để nhập các thông số
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Nhận dữ liệu từ form thông qua request.form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Đưa vào mô hình để dự đoán
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

        species = ['Setosa', 'Versicolor', 'Virginica']
        prediction = species[prediction]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
