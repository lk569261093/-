import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 读取数据
data = pd.read_excel('data.xlsx')
data = data.iloc[:, 1:9]  # 选择第二列到第八列的数据

# 2. 数据预处理
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])
    return np.array(X), np.array(Y)

look_back = 5
X, Y = create_dataset(data_scaled, look_back)

# 3. 创建LSTM模型并训练
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(7))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, Y, epochs=100, batch_size=1, verbose=1)

# 4. 预测2023年5月9日的数据
input_data = data_scaled[-look_back:]
input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))
prediction = model.predict(input_data)
predicted_values = scaler.inverse_transform(prediction)

print("2023年5月9日的预测结果：")
print(predicted_values)
