# SEBS 智能分析平台

基于 Flask + scikit-learn 的 **SEBS 材料性能预测 Web 服务**。上传实验数据 Excel 文件，自动完成数据清洗、多模型训练对比、参数优化与模型解释，生成 14 张可视化图表，帮助研究人员快速理解工艺参数对材料硬度的影响。

## 功能

- **数据预处理**：自动校验列名、解析质量比、缺失值处理、标准化
- **模型对比**：13 种回归模型（Linear / Ridge / Lasso / KNN / SVR / MLP / DecisionTree / ExtraTree / RandomForest / AdaBoost / Bagging / GradientBoosting），留一法交叉验证，输出 R² / MAE 饼图
- **参数优化**：GradientBoosting 学习率扫描，自动确定最佳参数
- **模型解释**：SHAP（条形图、蜂群图、瀑布图、依赖图、热力图、交互图）+ PDP / ICE 部分依赖图
- **前端交互**：拖拽上传、深色模式、图表全屏预览、单张下载、一键打包 ZIP

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python / Flask / Flask-CORS |
| 机器学习 | scikit-learn / XGBoost / CatBoost / SHAP |
| 数据处理 | pandas / numpy |
| 可视化 | matplotlib / seaborn |
| 前端 | 原生 HTML/CSS/JS + Axios + JSZip + FileSaver |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app.py
```

服务默认运行在 `http://localhost:5000`。

### 3. 使用

1. 浏览器打开 `http://localhost:5000`
2. 上传 Excel 文件（需包含列：`设计St:Bd:St质量比`、`硬度`、`断裂伸长率/%` 及工艺条件列）
3. 点击「开始分析」，等待 10-30 秒
4. 查看模型性能指标与可视化图表

## 项目结构

```
├── app.py              # Flask 后端（API + 分析逻辑）
├── index.html          # 前端页面
├── Jiegoutu.py         # 架构图生成脚本
├── Jiegou.html         # 可交互架构图
├── Jiegou_02.html      # 架构图变体
├── architecture_detailed.png  # 架构图（PNG）
├── requirements.txt    # Python 依赖
└── README.md
```

## Excel 数据格式要求

上传的 Excel 文件至少需包含以下列：

| 列名 | 说明 |
|------|------|
| `设计St:Bd:St质量比` | 质量比，格式如 `30:40:30` |
| `硬度` | 目标变量 |
| `断裂伸长率/%` | 用于筛选有效数据 |
| `单体浓度/%` 等 19 列工艺条件 | 作为模型特征 |
