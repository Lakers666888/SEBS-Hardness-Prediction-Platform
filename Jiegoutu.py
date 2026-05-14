import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import matplotlib

# 设置中文字体（Windows系统常用 SimHei，Linux/Mac 可尝试 'WenQuanYi Micro Hei' 或 'Noto Sans CJK SC'）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# 辅助函数：绘制带文字的矩形
def draw_rect(x, y, width, height, text, color='lightblue', text_size=9, align='center'):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha=align, va='center', fontsize=text_size, wrap=True)

# 辅助函数：绘制箭头
def draw_arrow(start, end, text=''):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# ==================== 用户层 ====================
draw_rect(0.5, 9.5, 15, 2.2, '用户层（浏览器）', color='#d9ead3', text_size=12)
# 前端具体模块
draw_rect(0.8, 10.0, 3.2, 1.5, '文件上传\n(点击/拖拽)\n支持 .xlsx/.xls', color='#fff2cc', text_size=9)
draw_rect(4.5, 10.0, 3.2, 1.5, '分析控制\n(开始/重置)\n深色模式', color='#fff2cc', text_size=9)
draw_rect(8.2, 10.0, 3.2, 1.5, '结果展示\n(指标卡片)\n14张图表', color='#fff2cc', text_size=9)
draw_rect(12.0, 10.0, 3.2, 1.5, '图表操作\n(全屏/下载)\n打包导出 ZIP', color='#fff2cc', text_size=9)

# ==================== 后端服务层 ====================
draw_rect(0.5, 0.5, 15, 8.5, '后端服务（Flask）', color='#f4cccc', text_size=12)

# 子模块布局
# 模块1：数据预处理
draw_rect(0.8, 1.2, 3.2, 3.5, '数据预处理', color='#daeef3', text_size=10)
draw_rect(0.9, 1.8, 3.0, 2.8, '• Excel 读取\n• 必需列校验\n• 质量比解析\n• 缺失值处理\n• 唯一值列删除\n• 标准化 (StandardScaler)', color='#f9f9f9', text_size=8, align='left')

# 模块2：模型训练与对比
draw_rect(4.5, 1.2, 3.2, 3.5, '模型训练与对比', color='#daeef3', text_size=10)
draw_rect(4.6, 1.8, 3.0, 2.8, '• 13种回归模型\n• 留一法交叉验证\n• R² & MAE 计算\n• 模型性能饼图生成', color='#f9f9f9', text_size=8, align='left')

# 模块3：参数优化
draw_rect(8.2, 1.2, 3.2, 3.5, '参数优化', color='#daeef3', text_size=10)
draw_rect(8.3, 1.8, 3.0, 2.8, '• 学习率扫描\n   (0.01~1.0)\n• 留一法验证\n• 生成 R² 折线图\n• 确定最佳学习率', color='#f9f9f9', text_size=8, align='left')

# 模块4：模型解释
draw_rect(12.0, 1.2, 3.2, 3.5, '模型解释', color='#daeef3', text_size=10)
draw_rect(12.1, 1.8, 3.0, 2.8, '• SHAP 系列图\n  (条形/蜂群/瀑布/依赖/热力/交互)\n• PDP 系列图\n  (一维/ICE/二维)\n• 特征重要性分析', color='#f9f9f9', text_size=8, align='left')

# 模块5：结果封装
draw_rect(0.8, 5.2, 14.4, 2.0, '结果封装', color='#daeef3', text_size=10)
draw_rect(0.9, 5.8, 14.2, 1.2, '将生成的图表转换为 base64 编码，连同性能指标打包为 JSON 返回前端', color='#f9f9f9', text_size=8, align='left')

# ==================== 数据流箭头 ====================
# 用户上传 -> 后端
draw_arrow((8, 9.5), (8, 5.8), 'POST /upload (Excel)')
# 后端返回 -> 前端
draw_arrow((8, 5.2), (8, 9.5), 'JSON (指标 + base64图片)')

# 模块间数据流（可选）
draw_arrow((2.5, 4.7), (6, 4.7), '特征矩阵\n目标变量')
draw_arrow((6, 4.7), (9.8, 4.7), '最佳模型')
draw_arrow((9.8, 4.7), (13.6, 4.7), 'SHAP/PDP 结果')
draw_arrow((13.6, 4.7), (8, 5.8), '图表列表')

# 添加标题和说明
plt.title('SEBS 智能分析平台总体架构', fontsize=18, pad=20)
ax.text(0.5, 11.5, '用户通过浏览器上传 Excel 文件，后端自动完成数据清洗、模型训练、解释分析，并返回可视化结果。', fontsize=10, ha='left')

plt.tight_layout()
plt.savefig('architecture_detailed.png', dpi=200, bbox_inches='tight')
plt.show()