import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

# ========== 离线模拟 Carseats 数据集（不用联网下载！） ==========
np.random.seed(0)
n=400
data = pd.DataFrame({
    "Sales": np.random.normal(8,3,n),
    "Price": np.random.uniform(50,150,n),
    "Income": np.random.uniform(20,80,n),
    "Advertising": np.random.uniform(0,20,n),
    "ShelveLoc": np.random.choice(["Bad","Medium","Good"], size=n)
})

# 自变量
X = data[["Price","Income","Advertising"]]
y = data["Sales"]

# 定性变量ShelveLoc转哑变量，删掉基准组
X = pd.get_dummies(X.join(data["ShelveLoc"]), columns=["ShelveLoc"], drop_first=True)
X = sm.add_constant(X)   # 添加常数项β0

# 拟合多元回归
model = sm.OLS(y, X).fit()
print(model.summary())

print("\n==== ShelveLoc基准组是 Bad ====")
print("ShelveLoc有三个取值：Bad, Medium, Good")
print('ShelveLoc[Good]系数解释：')
print("在其他变量不变时，货架位置Good相比Bad，销量平均变动多少\n")

# 计算VIF多重共线性检验
vif_df = pd.DataFrame()
vif_df["变量"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n===== VIF 多重共线性检验 =====")
print(vif_df)