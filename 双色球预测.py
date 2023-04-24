import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 读取Excel数据
data = pd.read_excel('data1.xlsx')

# 将日期列作为索引，并将其转换为时间序列数据类型
data.set_index('日期', inplace=True)
data.index = pd.to_datetime(data.index)

# 可视化原始数据
plt.figure(figsize=(10,5))
plt.plot(data['第二列'], label='第二列')
plt.plot(data['第三列'], label='第三列')
plt.plot(data['第四列'], label='第四列')
plt.plot(data['第五列'], label='第五列')
plt.plot(data['第六列'], label='第六列')
plt.plot(data['第七列'], label='第七列')
plt.plot(data['第八列'], label='第八列')
plt.legend()
plt.show()

# 构建特征矩阵和标签
X = np.array(range(len(data)))
y2 = data['第二列'].values
y3 = data['第三列'].values
y4 = data['第四列'].values
y5 = data['第五列'].values
y6 = data['第六列'].values
y7 = data['第七列'].values
y8 = data['第八列'].values

# 创建线性回归模型并训练
model2 = LinearRegression()
model2.fit(X.reshape(-1,1), y2)
model3 = LinearRegression()
model3.fit(X.reshape(-1,1), y3)
model4 = LinearRegression()
model4.fit(X.reshape(-1,1), y4)
model5 = LinearRegression()
model5.fit(X.reshape(-1,1), y5)
model6 = LinearRegression()
model6.fit(X.reshape(-1,1), y6)
model7 = LinearRegression()
model7.fit(X.reshape(-1,1), y7)
model8 = LinearRegression()
model8.fit(X.reshape(-1,1), y8)

# 预测2023年4月23日的数据
n_days = (pd.to_datetime('2023-04-23') - data.index[0]).days
pred_2 = model2.predict(np.array([len(data)+n_days]).reshape(-1,1))
pred_3 = model3.predict(np.array([len(data)+n_days]).reshape(-1,1))
pred_4 = model4.predict(np.array([len(data)+n_days]).reshape(-1,1))
pred_5 = model5.predict(np.array([len(data)+n_days]).reshape(-1,1))
pred_6 = model6.predict(np.array([len(data)+n_days]).reshape(-1,1))
pred_7 = model7.predict(np.array([len(data)+n_days]).reshape(-1,1))
pred_8 = model8.predict(np.array([len(data)+n_days]).reshape(-1,1))

# 输出预测结果
print('第二列预测值：', pred_2[0])
print('第三列预测值：', pred_3[0])
print('第四列预测值：', pred_4[0])
print('第五列预测值：', pred_5[0])
print('第六列预测值：', pred_6[0])
print('第七列预测值：', pred_7[0])
print('第八列预测值：', pred_8[0])
