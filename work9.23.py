import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# 生成模拟数据集，完全不需要联网！
np.random.seed(42)
n_samples = 300
n_features = 10
X = np.random.randn(n_samples, n_features)
true_beta = np.array([3, 2, 1.5, 0, 0, 0, 0, 0, 0, 0])
y = X @ true_beta + np.random.randn(n_samples)*0.8

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Z-score标准化（作业要求）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 候选lambda
alphas = np.logspace(-3, 3, 100)

# Ridge岭回归
ridge = RidgeCV(alphas=alphas, cv=10)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_coef = ridge.coef_
ridge_nonzero = np.sum(np.abs(ridge_coef) > 1e-8)

# Lasso
lasso = LassoCV(alphas=alphas, cv=10, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_coef = lasso.coef_
lasso_nonzero = np.sum(np.abs(lasso_coef) > 1e-8)

# ElasticNet
enet = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=10, random_state=42)
enet.fit(X_train_scaled, y_train)
enet_pred = enet.predict(X_test_scaled)
enet_rmse = np.sqrt(mean_squared_error(y_test, enet_pred))
enet_coef = enet.coef_
enet_nonzero = np.sum(np.abs(enet_coef) > 1e-8)

# 打印结果
print("="*50)
print(f"Ridge | 最优λ={ridge.alpha_:.4f}, 测试RMSE={ridge_rmse:.2f}, 非零变量数:{ridge_nonzero}")
print(f"Lasso | 最优λ={lasso.alpha_:.4f}, 测试RMSE={lasso_rmse:.2f}, 非零变量数:{lasso_nonzero}")
print(f"ElasticNet | 最优λ={enet.alpha_:.4f},  测试RMSE={enet_rmse:.2f}, 非零变量数:{enet_nonzero}")
print("="*50)

# 绘制系数路径图
coef_ridge = []
coef_lasso = []
coef_enet = []

for a in alphas:
    # Ridge
    ridge_temp = Ridge(alpha=a)
    ridge_temp.fit(X_train_scaled, y_train)
    coef_ridge.append(ridge_temp.coef_)
    # Lasso
    lasso_temp = Lasso(alpha=a, random_state=42)
    lasso_temp.fit(X_train_scaled, y_train)
    coef_lasso.append(lasso_temp.coef_)
    # ElasticNet
    enet_temp = ElasticNet(alpha=a, l1_ratio=0.5, random_state=42)
    enet_temp.fit(X_train_scaled, y_train)
    coef_enet.append(enet_temp.coef_)

coef_ridge = np.array(coef_ridge)
coef_lasso = np.array(coef_lasso)
coef_enet = np.array(coef_enet)

fig, axes = plt.subplots(1,3,figsize=(18,5))
# Ridge
axes[0].plot(alphas, coef_ridge)
axes[0].set_xscale("log")
axes[0].set_title("Ridge Coefficient Path")
axes[0].set_xlabel("λ(alpha)")
axes[0].set_ylabel("Coefficients")
# Lasso
axes[1].plot(alphas, coef_lasso)
axes[1].set_xscale("log")
axes[1].set_title("Lasso Coefficient Path")
axes[1].set_xlabel("λ(alpha)")
# ElasticNet
axes[2].plot(alphas, coef_enet)
axes[2].set_xscale("log")
axes[2].set_title("ElasticNet Coefficient Path")
axes[2].set_xlabel("λ(alpha)")

plt.tight_layout()
plt.show()
