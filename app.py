"""
SEBS 材料性能预测 Web 服务
将原始数据分析脚本封装为 Flask API，接收 Excel 文件，返回分析结果（图表 + 指标）
"""
import io
import base64
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# 回归模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor

# 预处理与评估
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# 模型解释
import shap
from sklearn.inspection import PartialDependenceDisplay

# 忽略警告
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # 允许跨域

# ------------------------- 辅助函数：保存 matplotlib 图表为 base64 -------------------------
def fig_to_base64(fig):
    """将 matplotlib Figure 对象转为 base64 编码的 PNG 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def save_current_fig(name, category):
    """保存当前 matplotlib 图形到全局 figures 列表"""
    fig = plt.gcf()
    img = fig_to_base64(fig)
    figures.append({
        "category": category,
        "name": name,
        "image": img
    })

# ------------------------- 核心分析函数 -------------------------
def run_analysis(file_bytes):
    """
    执行完整的数据分析流程
    参数: file_bytes (bytes) - Excel 文件内容
    返回: dict 包含 metrics 和 figures
    """
    global figures
    figures = []  # 用于存储所有图表

    # -------------------- 1. 读取数据并检查 --------------------
    sebs_data = pd.read_excel(io.BytesIO(file_bytes))
    print("读取成功，数据形状：", sebs_data.shape)
    print("列名：", list(sebs_data.columns))

    # 检查必需列
    required_cols = ['设计St:Bd:St质量比', '硬度', '断裂伸长率/%']
    missing = [col for col in required_cols if col not in sebs_data.columns]
    if missing:
        raise ValueError(f"Excel 缺少必需列：{missing}。请检查文件格式。")

    column_name = '断裂伸长率/%'
    sebs = sebs_data.dropna(subset=[column_name])
    print(f"筛选后数据行数：{len(sebs)}")

    if len(sebs) == 0:
        raise ValueError(f"列 '{column_name}' 所有值均为空，无法分析。")

    # 解析 St:Bd:St 质量比
    stbdst = []
    for i in range(len(sebs)):
        parts = sebs[['设计St:Bd:St质量比']].iloc[i, 0].split(":")
        st1, bd, st2 = map(float, parts)
        stbdst.append([st1, bd, st2])
    stbdst = pd.DataFrame(stbdst, columns=['st1', 'bd', 'st2'])

    # 工艺条件特征
    tiaojian_cols = ['单体浓度/%', '一段聚合温度/ºC', '二三段聚合温度/ºC',
                     'A釜预除杂n-BuLi/mL', 'B釜引发n-BuLi/mL', 'n-BuLi总量/mL',
                     'PS含量/%', '苯乙烯含量/wt%', '1,2-丁二烯含量/%', '嵌段比/%',
                     '镍mg/g', '铝镍比', '氢压/MPa', '加氢温度/℃', '加氢时间/h',
                     '核磁加氢度/%', '加氢分子量/万', '加氢PDI', '加氢胶液粘度/cP']
    tiaojian = sebs[tiaojian_cols]
    tiaojian.index = stbdst.index

    data_x = pd.concat([stbdst, tiaojian], axis=1)
    if data_x.empty:
        raise ValueError("特征数据为空，请检查输入文件。")

    data_x_n = data_x.loc[:, data_x.nunique() > 1]  # 删除唯一值列

    # 特征名称（英文）
    feature_names = [
        'st1', 'bd', 'st2',
        'Monomer concentration/%',
        'First-stage polymerization temperature/ºC',
        'Second and third-stage polymerization temperature/ºC',
        'Pre-impurity removal n-BuLi in A reactor/mL',
        'Initiation n-BuLi in B reactor/mL',
        'Total n-BuLi/mL',
        'PS content/%',
        'Styrene content/wt%',
        '1,2-butadiene content/%',
        'Block ratio/%',
        'Nickel mg/g',
        'Aluminum to nickel ratio',
        'Hydrogen pressure/MPa',
        'Hydrogenation degree by NMR/%',
        'Hydrogenated molecular weight/10,000',
        'Hydrogenated PDI',
        'Hydrogenated gel solution viscosity/cP'
    ]

    # 标准化
    scaler = StandardScaler()
    X = pd.DataFrame(data_x_n.fillna(0).values)
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # 目标变量：硬度（使用与 X_scaled 相同的 index）
    y = pd.DataFrame(sebs['硬度'].values, index=X_scaled.index, columns=['hardness'])

    # 完整数据集（用于可视化）
    X.columns = feature_names
    data_all = pd.concat([X, y], axis=1)

    # -------------------- 2. 数据探索可视化 --------------------
    sns.set(style="ticks")
    # 2.1 散点图矩阵（选取部分特征）
    vars_subset = ['Total n-BuLi/mL', 'PS content/%', '1,2-butadiene content/%',
                   'Block ratio/%', 'Nickel mg/g', 'Aluminum to nickel ratio',
                   'Hydrogenated gel solution viscosity/cP', 'hardness']
    pairplot = sns.pairplot(data_all, vars=vars_subset, diag_kind="hist", plot_kws={'alpha': 0.6})
    save_current_fig("散点图矩阵", "数据探索")

    # 2.2 相关性热力图
    plt.figure(figsize=(12, 8))
    correlation_matrix = data_all.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    save_current_fig("相关性热力图", "数据探索")

    # -------------------- 3. 多模型对比 --------------------
    models = [
        LinearRegression(), KNeighborsRegressor(), Ridge(), Lasso(),
        MLPRegressor(), DecisionTreeRegressor(), ExtraTreeRegressor(),
        AdaBoostRegressor(n_estimators=300, random_state=10),
        BaggingRegressor(), AdaBoostRegressor(),
        RandomForestRegressor(n_estimators=30, random_state=10),
        GradientBoostingRegressor(),
        SVR(kernel='rbf', C=100, gamma='auto')
    ]

    model_r2 = {}
    model_mae = {}
    for model in models:
        sp = LeaveOneOut()
        y_pred_list = []
        for train_idx, test_idx in sp.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx, 0], y.iloc[test_idx, 0]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
        r2 = r2_score(y, y_pred_list)
        model_r2[model.__class__.__name__] = max(r2, 0)
        model_mae[model.__class__.__name__] = mean_absolute_error(y, y_pred_list)

    # 饼图展示
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    r2_vals = list(model_r2.values())
    r2_labels = list(model_r2.keys())
    max_r2_idx = np.argmax(r2_vals)
    explode_r2 = [0.05 if i == max_r2_idx else 0 for i in range(len(r2_labels))]
    plt.pie(r2_vals, labels=r2_labels, autopct='%1.1f%%', explode=explode_r2, startangle=90, textprops={'fontsize': 8})
    plt.title('R² Score Distribution')
    plt.subplot(1, 2, 2)
    mae_vals = list(model_mae.values())
    mae_labels = list(model_mae.keys())
    min_mae_idx = np.argmin(mae_vals)
    explode_mae = [0.05 if i == min_mae_idx else 0 for i in range(len(mae_labels))]
    plt.pie(mae_vals, labels=mae_labels, autopct='%1.1f%%', explode=explode_mae, startangle=90, textprops={'fontsize': 8})
    plt.title('MAE Score Distribution')
    plt.tight_layout()
    save_current_fig("模型性能饼图 (R² & MAE)", "模型对比")

    # -------------------- 4. 学习率优化折线图（新增） --------------------
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    n_estimators_fixed = 100
    max_depth_fixed = 3
    r2_by_lr = []
    for lr in learning_rates:
        model_temp = GradientBoostingRegressor(
            n_estimators=n_estimators_fixed, learning_rate=lr, max_depth=max_depth_fixed,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        loo = LeaveOneOut()
        y_pred_list = []
        for train_idx, test_idx in loo.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx, 0], y.iloc[test_idx, 0]
            model_temp.fit(X_train, y_train)
            y_pred_list.append(model_temp.predict(X_test))
        r2 = r2_score(y, y_pred_list)
        r2_by_lr.append(max(r2, 0))

    plt.figure(figsize=(12, 6))
    plt.plot(learning_rates, r2_by_lr, marker='o', linestyle='-', linewidth=2, markersize=8,
             color='#1f77b4', markerfacecolor='red')
    best_lr = learning_rates[np.argmax(r2_by_lr)]
    best_r2 = max(r2_by_lr)
    plt.scatter([best_lr], [best_r2], color='red', s=100, edgecolors='black', zorder=5)
    plt.annotate(f'Best: LR={best_lr}\nR²={best_r2:.4f}',
                 xy=(best_lr, best_r2), xytext=(best_lr+0.1, best_r2-0.05),
                 fontsize=12, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.xlabel('Learning Rate')
    plt.ylabel('R² Score')
    plt.title('GradientBoosting Performance vs Learning Rate')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xscale('log')
    plt.xticks(learning_rates, [str(lr) for lr in learning_rates], rotation=45)
    for i, (lr, r2_val) in enumerate(zip(learning_rates, r2_by_lr)):
        plt.text(lr, r2_val + 0.01, f'{r2_val:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    save_current_fig("学习率优化曲线", "参数优化")

    # -------------------- 5. 最优模型预测效果 --------------------
    best_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.5, max_depth=3,
        min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    y_true_list = []
    y_pred_list = []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx, 0], y.iloc[test_idx, 0]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        # 确保 y_test 是标量（如果还是 Series 则提取值）
        if hasattr(y_test, 'values') and len(y_test) == 1:
            y_test = y_test.values[0]
        # y_pred 是 ndarray，取第一个元素
        if isinstance(y_pred, np.ndarray) and len(y_pred) == 1:
            y_pred = y_pred[0]
        y_true_list.append(float(y_test))
        y_pred_list.append(float(y_pred))
    avg_mae = mean_absolute_error(y_true_list, y_pred_list)
    avg_r2 = r2_score(y_true_list, y_pred_list)

    plt.figure(figsize=(10, 7))
    plt.scatter(y_true_list, y_pred_list, color='blue', alpha=0.6, s=85)
    lim = [min(y_true_list) - 5, max(y_true_list) + 5]
    plt.plot(lim, lim, 'k--', lw=2)
    plt.title('True vs Predicted Hardness')
    plt.xlabel('True Hardness')
    plt.ylabel('Predicted Hardness')
    plt.xlim(lim)
    plt.ylim(lim)
    plt.text(lim[1]*0.85, lim[1]*0.55, f'MAE: {avg_mae:.2f}\nR²: {avg_r2:.2f}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.grid()
    plt.tight_layout()
    save_current_fig("真实值 vs 预测值散点图", "模型验证")

    # -------------------- 6. SHAP 模型解释 --------------------
    best_model.fit(X_scaled, y)
    def model_wrapper(X):
        return best_model.predict(X)
    explainer = shap.Explainer(model_wrapper, X_scaled)
    shap_values = explainer(X_scaled)

    # 条形图
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=20, show=False)
    save_current_fig("SHAP 特征重要性条形图", "模型解释")

    # 蜂群图
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    save_current_fig("SHAP 蜂群图", "模型解释")

    # 单个样本瀑布图（取第一个样本）
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=20, show=False)
    save_current_fig("SHAP 瀑布图 (样本0)", "模型解释")

    # 特征依赖图（以1,2-丁二烯含量为例）
    feature = "1,2-butadiene content/%"
    interaction_feature = "Hydrogenated molecular weight/10,000"
    plt.figure()
    shap.plots.scatter(shap_values[:, feature], color=shap_values[:, interaction_feature], x_jitter=1.0, show=False)
    save_current_fig(f"SHAP 依赖图 ({feature})", "模型解释")

    # 热力图
    plt.figure(figsize=(12, 6))
    shap.plots.heatmap(shap_values, max_display=20, show=False)
    save_current_fig("SHAP 热力图", "模型解释")

    # 交互作用图
    shap_interaction = shap.TreeExplainer(best_model).shap_interaction_values(X_scaled)
    plt.figure()
    shap.dependence_plot((feature, interaction_feature), shap_interaction, X_scaled, feature_names=feature_names, show=False)
    save_current_fig(f"特征交互作用 ({feature} × {interaction_feature})", "模型解释")

    # -------------------- 7. PDP 部分依赖图 --------------------
    common_params = {"subsample": 50, "n_jobs": 2, "grid_resolution": 20, "random_state": 0}
    features_info = {"features": [feature, interaction_feature], "kind": "average"}
    _, ax = plt.subplots(ncols=2, figsize=(14, 4), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(best_model, X_scaled, **features_info, ax=ax, **common_params)
    plt.suptitle("Partial Dependence Plot")
    save_current_fig("一维 PDP 图", "模型解释")

    # ICE 曲线
    features_info_ice = {"features": [feature, interaction_feature], "kind": "both", "centered": True}
    _, ax = plt.subplots(ncols=2, figsize=(14, 4))
    PartialDependenceDisplay.from_estimator(best_model, X_scaled, **features_info_ice, ax=ax, **common_params)
    plt.suptitle("ICE 与 PDP 曲线")
    save_current_fig("ICE 个体条件期望图", "模型解释")

    # 二维 PDP
    features_info_2d = {"features": [(feature, interaction_feature)], "kind": "average"}
    display = PartialDependenceDisplay.from_estimator(best_model, X_scaled, **features_info_2d, **common_params)
    plt.suptitle("2D Partial Dependence Plot")
    save_current_fig("二维 PDP 等高线图", "模型解释")

    # -------------------- 8. 收集指标 --------------------
    metrics = {
        "best_model": best_model.__class__.__name__,
        "best_r2": round(avg_r2, 4),
        "best_mae": round(avg_mae, 4),
        "n_models": len(models)
    }

    return {"metrics": metrics, "figures": figures}


# ------------------------- Flask 路由 -------------------------
@app.route('/')
def index():
    """托管前端页面"""
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """接收 Excel 文件并返回分析结果"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "未上传文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "文件名为空"}), 400

    try:
        file_bytes = file.read()
        result = run_analysis(file_bytes)
        return jsonify({"status": "success", **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)