import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
methods = [
    'LLMParser',
    'AEL',
    'DRAIN', 
    'LibreLog',
    'PEAC-Log(TreeCache)',
    'PEAC-Log',
]

throughput = [
    48.39,
    11085.34,
    12054.23,
    19041.02,
    182160.01,  # PEAC-Log(TreeCache)
    123268.25,  # PEAC-Log
]

# 颜色配置（保持与原图相似）
colors = [
    '#6699cc',  # LLMParser: 蓝色
    '#7ccc7c',  # AEL: 绿色
    '#ff9966',  # DRAIN: 橙色
    '#cc66cc',  # LibreLog: 紫色
    '#9966cc',  # PEAC-Log(TreeCache): 深紫色
    '#4477aa',  # PEAC-Log: 深蓝色
]

# 创建图表
fig, ax = plt.subplots(figsize=(10, 5))

# 横向条形图
bars = ax.barh(methods, throughput, color=colors, edgecolor='white', linewidth=1)

# 设置对数刻度
ax.set_xscale('log')
ax.set_xlim(10, 200000)

# 添加数据标签（调整位置，避免遮挡）
for bar in bars:
    width = bar.get_width()
    # 根据数值大小调整标签位置
    if width < 1000:
        label_x = width * 1.1  # 小数值稍微偏移
    else:
        label_x = width * 1.01  # 大数值靠近条形
    ax.text(label_x, bar.get_y() + bar.get_height()/2,
            f'{width:,.2f}', va='center', fontsize=10, color='#333333')

# 设置标题和标签
ax.set_xlabel('Average Throughput (Logs/s) [Log Scale]', fontsize=12, labelpad=15)
ax.set_title('Comparison of Log Parsing Methods', fontsize=14, pad=20)

# 仅保留横线网格（移除竖线）
ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='#cccccc')

# 隐藏顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置坐标轴颜色
ax.spines['left'].set_color('#999999')
ax.spines['bottom'].set_color('#999999')

# 设置刻度样式
ax.tick_params(axis='both', which='major', labelsize=10, color='#666666')

# 调整布局
plt.tight_layout()

# 保存图片
output_path = Path(__file__).resolve().parent / 'throughput_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'图表已保存到: {output_path}')

# 显示图表
plt.show()
