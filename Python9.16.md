第1题 OLS最小二乘推导
一元线性回归模型：$\hat y_i=\hat\beta_0+\hat\beta_1 x_i$

残差：$e_i=y_i-\hat y_i$
残差平方和：$SSE=\sum (y_i-\hat\beta_0-\hat\beta_1 x_i)^2$

OLS目标：最小化残差平方和。
对$\hat\beta_0,\hat\beta_1$求偏导并令导数为0，解得：

$$
\hat\beta_1=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2},\quad \hat\beta_0=\bar y-\hat\beta_1\bar x
$$

OLS基本假设：线性、零条件均值、同方差、无自相关、无完全多重共线性。

---

第2题 $R^2$ 与调整$R^2$
平方和分解：$\boldsymbol{SST=SSR+SSE}$
- SST：总平方和
- SSR：回归平方和
- SSE：残差平方和

$R^2= \dfrac{SSR}{SST}=1-\dfrac{SSE}{SST}$
含义：自变量解释因变量波动的比例。
缺陷：增加变量时$R^2$不会减小，容易虚高。

调整$R^2=1-\dfrac{SSE/(n-k-1)}{SST/(n-1)}$
加入自由度惩罚，变量变多但解释力不足时，调整$R^2$会下降，可用于对比变量数不同的模型。