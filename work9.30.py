import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ====================== 1. 模拟Netflix用户-电影评分稀疏数据（对应Netflix Prize） ======================
np.random.seed(42)
n_samples = 2000  # 用户样本
n_features = 100   # 电影特征（高维 p>n场景，课件重点）
# 生成带共线性的特征，模拟用户观影偏好（贴合课件共线性问题）
X = np.random.randn(n_samples, n_features)
X[:, 10:20] = X[:, 0:10] + np.random.randn(n_samples,10)*0.05
# 真实评分，带噪声
beta_true = np.zeros(n_features)
beta_true[:15] = np.array([3, 2.8, 2.5, 2.2, 2, 1.8, 1.5, 1.2, 1, 0.9,0.7,0.6,0.5,0.4,0.3])
y = X @ beta_true + np.random.randn(n_samples)*0.4

# 构建DataFrame
df = pd.DataFrame(X, columns=[f"movie_{i}" for i in range(n_features)])
df["rating"] = y

# ====================== 2. 数据预处理 ======================
X = df.drop("rating", axis=1)
y = df["rating"]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化！正则化回归必须标准化（课件知识点）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== 3. ElasticNet 网格搜索调参（GridSearchCV） ======================
# alpha：正则强度；l1_ratio：L1占比，0=Ridge，1=Lasso，中间就是ElasticNet
param_grid = {
    "alpha": [0.01, 0.1, 0.5, 1, 5, 10],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}
elastic = ElasticNet(random_state=42)
grid_search = GridSearchCV(elastic, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X_train_scaled, y_train)

print("最优超参数：", grid_search.best_params_)
best_model = grid_search.best_estimator_

# ====================== 4. 模型预测与评估 ======================
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")

# ====================== 5. 系数可视化（课件重点：特征筛选） ======================
coef = best_model.coef_
plt.figure(figsize=(10,4))
plt.bar(range(len(coef)), coef)
plt.title("ElasticNet 特征系数（Netflix电影特征）")
plt.xlabel("电影特征编号")
plt.ylabel("系数大小，0代表被筛除")
plt.tight_layout()
plt.savefig("elasticnet_coef.png", dpi=300)
plt.show()

print("非零系数特征数量：", np.sum(np.abs(coef) > 1e-5))
