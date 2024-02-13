import numpy as np
import seaborn as sns
import pandas as pd


def main():
    model_data = sns.load_dataset('titanic')
    # model_data = pd.read_csv('')
    model_data = model_data.dropna()
    categorical_columns = ['class', 'sex', 'age', 'embark_town']
    target_column = 'survived'
    new_model_data = map_categorical_to_numeric(model_data, categorical_columns, target_column)
    print('model_data:\n', new_model_data)
    print(new_model_data.dtypes)
    columns_for_deviation = ['class', 'sex', 'age', 'embark_town']
    deviation_model = calculate_deviation_model(new_model_data, columns_for_deviation)
    new_data = pd.DataFrame({'class': ['First'], 'sex': ['female'], 'age': [25], 'embark_town': ['Southampton']})
    new_data_dev = calculate_deviation_for_new_data(new_data, deviation_model, new_model_data, columns_for_deviation, categorical_columns)
    print('new_data_dev:\n', new_data_dev)

def calculate_z_score(value, mean, std_dev):
    if std_dev == 0:
        return 0
    else:
        print(f"Z_score=\n({value} - {mean} / {std_dev})")
        return (value - mean) / std_dev


def calculate_deviation_score(value, mean, std_dev, base_value=50):
    z_score = calculate_z_score(value, mean, std_dev)
    deviation_score = z_score * 10 + base_value
    print(f"{z_score} * 10 + {base_value}")
    print(f"deviation_score: {deviation_score}")
    return deviation_score


def calculate_mean_and_std_dev(data):
    mean_value = np.mean(data)
    std_dev_value = np.std(data)
    return mean_value, std_dev_value


def map_categorical_to_numeric(data, columns, target_column):
    for col in columns:
        if data[col].dtype != 'category':
            data[col] = data[col].astype('category')
        # カテゴリカル列のマッピングを作成
        encoding_mapping = data[[col, target_column]].groupby([col])[target_column].mean().to_dict()
        print('encoding_mapping:\n', encoding_mapping)
        # カテゴリカル列を数値に変換
        data[col + '_mean'] = data[col].map(encoding_mapping)
        print('mapping_data:\n', data)
        # もしデータ型が 'category' なら整数に変換
        if data[col + '_mean'].dtype.name == 'category':
            data[col + '_mean'] = data[col + '_mean'].astype('float64')

    return data

def calculate_deviation_model(data, columns_for_deviation):
    deviation_model = {}  # ダミーのデータフレームを作成

    for column in columns_for_deviation:
        mean_value, std_dev_value = calculate_mean_and_std_dev(data[column + '_mean'])
        deviation_model[column + '_model'] = [mean_value, std_dev_value]
    print(deviation_model)
    return deviation_model


def calculate_deviation_for_new_data(new_data, deviation_model, model_data, columns_for_deviation, categorical_columns):
    # 新しいデータのカテゴリカルデータを数値データに変更
    new_data = preprocess_new_data(new_data, categorical_columns, model_data)
    # print(new_data)
    # 新しいデータに偏差値情報を追加
    print('start deviation')
    for col in columns_for_deviation:
        mean = deviation_model[col + '_model'][0]
        std_dev = deviation_model[col + '_model'][1]
        new_data[col + '_deviation'] = calculate_deviation_score(new_data[col].values[0], mean, std_dev, 50)
        print('=============')
        print(new_data[col + '_deviation'])
    return new_data


# 新しいデータの前処理
def preprocess_new_data(new_data, categorical_columns, model_data):
    for col in categorical_columns:
        print('model_data[col + _mean]')
        print(model_data[col + '_mean'])
        # モデルのカテゴリと新しいデータの値が一致する場合、対応する数値データを取得し、異なる場合は0を設定
        df_dict = dict(zip(model_data[col], model_data[col + '_mean'].values))
        print(f"df_dict= {df_dict}")
        new_data[col] = new_data[col].map(lambda x: df_dict.get(x, 0))

    return new_data


if __name__ == '__main__':
    main()
