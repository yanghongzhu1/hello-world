import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from ucimlrepo import fetch_ucirepo
warnings.filterwarnings("ignore")

# 加载spambase数据集（UCI原版，和Kaggle spambase数据集完全相同，无需手动下载csv）
spambase = fetch_ucirepo(id=94)
X = spambase.data.features
y = spambase.data.targets.values.ravel()

# 2.特征标准化（SVM重点：对特征尺度非常敏感，课件强调）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3.划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4.网格搜索GridSearchCV，调超参数C和gamma（RBF核）
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("最优超参数：", grid.best_params_)
model = grid.best_estimator_

# 5.预测评估
y_pred = model.predict(X_test)
print("测试集准确率：", accuracy_score(y_test, y_pred))
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['正常邮件','垃圾邮件'])
plt.yticks(tick_marks, ['正常邮件','垃圾邮件'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()
