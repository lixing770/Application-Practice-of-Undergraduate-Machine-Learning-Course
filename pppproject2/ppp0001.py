import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.dates as mdates

# 1. 导入训练数据
train_data = pd.read_csv('total.csv')  # 替换为你的文件路径

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 常用 Mac 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号时避免乱码

# 2. 提取 'TimeInterval' 列中的起始时间并转换为 datetime 类型
train_data['StartTime'] = train_data['TimeInterval'].apply(lambda x: '-'.join(x.split('-')[:3]))  # 提取日期部分
train_data['StartTime'] = pd.to_datetime(train_data['StartTime'], format='%Y-%m-%d %H:%M:%S')  # 转换为 datetime 类型

# 3. 特征工程：选择特征和标签
X = train_data[['GridID', 'StartTime']]  # 特征，网格ID和时间
X['Hour'] = X['StartTime'].dt.hour  # 提取小时
X['DayOfWeek'] = X['StartTime'].dt.dayofweek  # 提取星期几
X['Month'] = X['StartTime'].dt.month  # 提取月份
X = X[['GridID', 'Hour', 'DayOfWeek', 'Month']]  # 使用这些特征

y = train_data['DataCount']  # 标签：打车需求量

# 4. 数据集划分：训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 线性回归模型
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)

# 6. 多项式回归模型
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly_train, y_train)
y_pred_poly = poly_regressor.predict(X_poly_test)

# 7. 神经网络回归模型
nn_model = Sequential()
nn_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1))  # 输出层
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
y_pred_nn = nn_model.predict(X_test)

# 将神经网络的预测结果转换为一维数组
y_pred_nn = y_pred_nn.flatten()

# 8. 评估各模型：计算R², MSE, RMSE, MAE
def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mse, rmse, mae

# 评估结果
r2_linear, mse_linear, rmse_linear, mae_linear = evaluate_model(y_test, y_pred_linear)
r2_poly, mse_poly, rmse_poly, mae_poly = evaluate_model(y_test, y_pred_poly)
r2_nn, mse_nn, rmse_nn, mae_nn = evaluate_model(y_test, y_pred_nn)

# 打印各模型的评估指标
print("线性回归模型指标:")
print(f"R²: {r2_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")
print(f"MAE: {mae_linear:.4f}")
print()

print("多项式回归模型指标:")
print(f"R²: {r2_poly:.4f}")
print(f"MSE: {mse_poly:.4f}")
print(f"RMSE: {rmse_poly:.4f}")
print(f"MAE: {mae_poly:.4f}")
print()

print("神经网络回归模型指标:")
print(f"R²: {r2_nn:.4f}")
print(f"MSE: {mse_nn:.4f}")
print(f"RMSE: {rmse_nn:.4f}")
print(f"MAE: {mae_nn:.4f}")
print()

# 9. 绘图：预测值与真实值对比图
def plot_prediction_comparison(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='真实值', color='blue', linestyle='-', marker='o', markersize=3)
    plt.plot(y_pred, label='预测值', color='red', linestyle='--', marker='x', markersize=5)
    plt.title(f'{model_name} - 预测值 vs 真实值')
    plt.xlabel('样本')
    plt.ylabel('需求量 (DataCount)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制各个模型的预测值与真实值对比图
plot_prediction_comparison(y_test, y_pred_linear, '线性回归')
plot_prediction_comparison(y_test, y_pred_poly, '多项式回归')
plot_prediction_comparison(y_test, y_pred_nn, '神经网络回归')

# 10. 绘图：残差分析图
def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), residuals, color='purple', s=10)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title(f'{model_name} - 残差分析')
    plt.xlabel('样本')
    plt.ylabel('残差')
    plt.grid(True)
    plt.show()

# 绘制各个模型的残差分析图
plot_residuals(y_test, y_pred_linear, '线性回归')
plot_residuals(y_test, y_pred_poly, '多项式回归')
plot_residuals(y_test, y_pred_nn, '神经网络回归')

# 11. 绘图：各模型性能对比图
def plot_performance_comparison(models, r2_scores, mse_scores, rmse_scores, mae_scores):
    x = np.arange(len(models))  # 模型索引
    width = 0.2  # 每个柱子的宽度

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - 1.5 * width, r2_scores, width, label='R²', color='lightblue')
    ax.bar(x - 0.5 * width, mse_scores, width, label='MSE', color='lightgreen')
    ax.bar(x + 0.5 * width, rmse_scores, width, label='RMSE', color='lightcoral')
    ax.bar(x + 1.5 * width, mae_scores, width, label='MAE', color='lightskyblue')

    ax.set_xlabel('模型')
    ax.set_ylabel('评分')
    ax.set_title('各模型性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.show()

# 模型名称和性能指标
models = ['线性回归', '多项式回归', '神经网络回归']
r2_scores = [r2_linear, r2_poly, r2_nn]
mse_scores = [mse_linear, mse_poly, mse_nn]
rmse_scores = [rmse_linear, rmse_poly, rmse_nn]
mae_scores = [mae_linear, mae_poly, mae_nn]

# 绘制性能对比图
plot_performance_comparison(models, r2_scores, mse_scores, rmse_scores, mae_scores)
