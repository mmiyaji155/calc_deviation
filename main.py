from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import pandas as pd
from calc import calculate_deviation_model, calculate_deviation_for_new_data, map_categorical_to_numeric


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return redirect(request.url)

        file = request.files['csv_file']
        if file.filename == '':
            return render_template('index.html', error='ファイルを選択してください')

        if file and allowed_file(file.filename):
            filename = 'model_data.csv'
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            return render_template('index.html', success='ファイルがアップロードされました')

    return render_template('index.html')


@app.route('/new_data', methods=['POST'])
def receive_new_data():
    try:
        # 保存されたCSVファイルを読み込む
        filename = 'model_data.csv'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        model_data = pd.read_csv(file_path)
        model_data = model_data.dropna()

        categorical_columns = ['age', 'history', 'number_of_companies', 'industry', 'occupation']
        target_column = ['fee', 'income']
        new_model_data = map_categorical_to_numeric(model_data, categorical_columns, target_column)
        columns_for_deviation = ['age', 'history', 'number_of_companies', 'industry', 'occupation']
        deviation_model = calculate_deviation_model(new_model_data, columns_for_deviation)

        # パラメータから各値を受け取る
        age = int(request.form['age'])
        history = request.form['history']
        number_of_companies = int(request.form['number_of_companies'])
        industry = request.form['industry']
        occupation = request.form['occupation']
        income = int(request.form['income'])

        # 新しいデータを成形
        new_data = pd.DataFrame({
            'age': [age],
            'history': [history],
            'number_of_companies': [number_of_companies],
            'industry': [industry],
            'occupation': [occupation],
            'income': [income]
        })

        # 新しいデータを偏差計算関数に渡す
        new_data_dev = calculate_deviation_for_new_data(new_data, deviation_model, new_model_data,
                                                        columns_for_deviation, categorical_columns)

        # 結果をJSON形式で返す
        return jsonify({'result': new_data_dev.to_dict(orient='records')})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
