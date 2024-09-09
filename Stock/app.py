from flask import Flask, render_template
import supplydata as sd

# Flask 앱 생성
app = Flask(__name__)


@app.route('/')
def predict():
  # 예측 수행
    stockprediction = sd.predicted_value()
    prediction_text = f'Predicted value: {stockprediction}'

    picture_path = sd.plot_picture()

    result_string = sd.get_result_string()

    return render_template('index.html', prediction=prediction_text, picture=picture_path, prediction_string=result_string)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug=True)