import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体（Mac 系统上的一个例子）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 常用 Mac 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号时避免乱码

# 1. 导入训练数据
train_data = pd.read_csv('total.csv')  # 替换为你的文件路径

# 查看数据的前几行，确认文件成功导入
print(train_data.head())

# 2. 提取 'TimeInterval' 列中的起始时间并转换为 datetime 类型
train_data['StartTime'] = train_data['TimeInterval'].apply(lambda x: '-'.join(x.split('-')[:3]))

# 将 'StartTime' 列转换为 datetime 类型
train_data['StartTime'] = pd.to_datetime(train_data['StartTime'], format='%Y-%m-%d %H:%M:%S')
# 查看转换后的数据
print(train_data[['TimeInterval', 'StartTime']])

# 3. 计算每个网格在每个时间点的打车需求量
grid_time_demand = train_data.groupby(['GridID', 'StartTime']).agg({'DataCount': 'sum'}).reset_index()

# 4. 找出 top-5 高频需求网格
total_demand_per_grid = train_data.groupby('GridID').agg({'DataCount': 'sum'}).reset_index()
top_5_grids = total_demand_per_grid.sort_values(by='DataCount', ascending=False).head(5)
top_5_grid_ids = top_5_grids['GridID'].tolist()

# 5. 获取 top-5 网格的需求数据
top_5_data = grid_time_demand[grid_time_demand['GridID'].isin(top_5_grid_ids)]

# 6. 绘制打车需求随时间变化的曲线图
plt.figure(figsize=(14, 6))  # 增加图形宽度

# 为每个 top-5 网格绘制一条曲线
for grid_id in top_5_grid_ids:
    grid_data = top_5_data[top_5_data['GridID'] == grid_id]
    if not grid_data.empty:
        plt.plot(grid_data['StartTime'], grid_data['DataCount'], label=f'Grid {grid_id}')

# 设置图形标题和标签
plt.title('Top-5 高频需求网格的打车需求随时间变化')
plt.xlabel('时间区间')
plt.ylabel('需求量 (DataCount)')

# 设置 X 轴格式，显示日期格式并减少密度
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 只显示日期，不显示时分秒
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 每天一个刻度

# 或者你可以使用自动间隔来优化标签
# plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

# 自动调整 x 轴标签，旋转45度避免重叠
plt.xticks(rotation=45)

# 调整布局，避免标签被遮挡
plt.subplots_adjust(bottom=0.2, top=0.9)

# 调整图例的位置，这里将图例放置在图表的右侧外面
plt.legend(title="网格ID", loc='upper left', bbox_to_anchor=(1, 1))

# 显示图形
plt.show()
