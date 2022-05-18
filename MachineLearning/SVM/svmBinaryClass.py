import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC

#导入数据
path = "data/cupcake or muffin.xlsx"
data = pd.read_excel(path)
# print(data.shape)#(18, 3)
# print(data)

#数据可视化
# sns.lmplot(data=data,x='Sugar',y='Butter',palette='Set1',fit_reg=False,hue='CakeType',scatter_kws={'s':150})
'''
lmplot()参数说明：
palette='Set1'设置调色板型号，对应不同绘图风格，色彩搭配。
fit_reg=False表示不显示拟合的回归线。因为lmplot()本身是线性回归绘图函数，默认会绘制点的拟合回归线。
hue='CakeType'表示对样本点按照'CakeType'的取值不同进行分类显示，这样不同类型的蛋糕会用不同颜色显示。若不设置hue参数，则所有点都会显示为一个颜色显示。
scatter_kws={'s':150}：设置点的大小，其中s表示size。
'''


#数据预处理
#将CakeType的值映射到0、1，方便后续模型运算
label = np.where(data['CakeType']=='muffin',0,1)
print(label)#[0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
x = data[['Sugar','Butter']]
# print(x)

#SVM实例化
#SVC指Support Vector Classifier
svc = SVC(kernel='linear',C=1)
'''
SVC参数说明：
C:惩罚系数，即当分类器错误地将A类样本划分为B类了，我们将给予分类器多大的惩罚。当我们给与非常大的惩罚，即C的值设置的很大，那么分类器会变得非常精准，但是，会产生过拟合问题。
kernel：核函数，如果使用一条直线就可以将属于不同类别的样本点全部划分开，那么我们使用kernel='linear'，
如果不能线性划分开，尤其是当数据维度很多时，一般很难找到一条合适的线将不同的类别的样本划分开，那么就尝试使用高斯核函数（也称为径向基核函数-rbf）、多项式核函数（poly）
'''
svc.fit(X=x,y=label)


#根据拟合结果，找出超平面
w = svc.coef_[0]
a = -w[0]/w[1]#超平面的斜率，也是边界线的斜率
xx = np.linspace(5,30)#生成5~30之间的50个数
#print(xx)
yy = a * xx - (svc.intercept_[0])/w[1]

#根据超平面，找到超平面的两条边界线
b = svc.support_vectors_[0]
yy_down = a * xx + (b[1]-a*b[0])
b = svc.support_vectors_[-1]
yy_up = a * xx + (b[1]-a*b[0])

#绘制超平面和边界线
#(1)绘制样本点的散点图
sns.lmplot(data=data,x='Sugar',y='Butter',hue='CakeType',palette='Set1',fit_reg=False,scatter_kws={'s':150})
#（2）向散点图添加超平面
plt.plot(xx,yy,linewidth=4,color='black')

#（3）向散点图添加边界线
plt.plot(xx,yy_down,linewidth=2,color='blue',linestyle='--')
plt.plot(xx,yy_up,linewidth=2,color='blue',linestyle='--')

plt.show()