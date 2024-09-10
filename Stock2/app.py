import os
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import supplydata as sd
import SVD_compression

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['STATIC_FOLDER'] = 'static/'

# 파일 삭제 함수
def delete_existing_files():
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.png')
    compressed_image_path = os.path.join(app.config['STATIC_FOLDER'], 'compressed_data.png')
    
    # input_image.png가 존재하면 삭제
    if os.path.exists(input_image_path):
        os.remove(input_image_path)
    
    # compressed_data.png가 존재하면 삭제
    if os.path.exists(compressed_image_path):
        os.remove(compressed_image_path)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/stock')
def predict():
  # 예측 수행
    stockprediction = sd.predicted_value()
    prediction_text = f"Predicted value: {stockprediction}"

    picture_path = sd.plot_picture()

    result_string = sd.get_result_string()

    return render_template('stock.html', prediction=prediction_text, picture=picture_path, prediction_string=result_string)

@app.route('/imageCompression', methods=['GET', 'POST'])
def upload_file():
    original_image_name = None
    compressed_image_filename = None  # 압축된 이미지 파일명

    if request.method == 'POST':
        # 기존 파일 삭제
        delete_existing_files()

        file = request.files['file']
        if file and allowed_file(file.filename):
            # 파일 이름을 무조건 'input_image.png'로 설정
            filename = 'input_image.png'
            original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_file_path)
            image = Image.open(original_file_path)

            image_data = np.array(image)
            red_ratio = float(request.form.get('red_ratio', 0.10))
            green_ratio = float(request.form.get('green_ratio', 0.10))
            blue_ratio = float(request.form.get('blue_ratio', 0.10))

            compressed_image = SVD_compression.compress_image(image_data, red_ratio, green_ratio, blue_ratio)
            SVD_compression.SVD_picture_path(compressed_image, output_path=os.path.join(app.config['STATIC_FOLDER'], 'compressed_data.png'))

    compressed_image_file = url_for('static', filename='compressed_data.png')
    original_image_name = url_for('uploaded_file', filename='input_image.png')  # uploads 폴더에서 파일 경로를 가져옴

    return render_template('imageCompression.html', original_image_file=original_image_name, compressed_image_file=compressed_image_file)

# 파일 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# 사용자가 업로드한 파일을 제공할 수 있는 라우트 정의
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
