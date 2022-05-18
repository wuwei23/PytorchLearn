import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC

#导入数据
path = "data/cupcake and muffin and Cavine.xlsx"
data = pd.read_excel(path)
# print(data)
# print(data.shape)#(24, 3)
print(data['CakeType'].value_counts())


#数据可视化
# sns.lmplot(data=data,x='Sugar',y='Butter',palette='Set1',fit_reg=False,hue='CakeType',scatter_kws={'s':150})

#数据预处理
#将CakeType的值映射到0、1、2，方便后续模型运算
label = data.CakeType.map({'muffin' : 0, 'cupcake' : 1, 'Cavine' : 2})
# print(label)
x = data[['Sugar','Butter']]


#实例化SVC
svc = SVC(kernel='linear',C=0.1, decision_function_shape='ovr')
'''
参数说明：
decision_function_shape：设置分类器决策模式，有两种模式：ovr和ovo
ovr表示one vs rest模式，即比较某一个类别和其他所有类别；
ovo表示one vs one模式，即一对一地比较（两两比较）两个类别，如果样本数据有ABC3个类别，则需要比较3次，分别是比较A和B，比较A和C，比较B和C。
ovo模式需要比较的次数会多于ovr模式，但是，ovo模式对于类别不均衡的样本数据具有较好的稳定性。
'''
svc.fit(X=x,y=label)
label_predict = svc.predict(X=x)


#绘制三分类超平面和边界线
#生成坐标
x_min, x_max = data.Sugar.min() - 0.2, data.Sugar.max() + 0.2
y_min, y_max = data.Butter.min() - 0.2, data.Butter.max() + 0.2
step = 0.01
x_value = np.arange(x_min, x_max, step)
y_value = np.arange(y_min, y_max, step)
#注意！因为x_min到x_max的距离可能并不等于y_min到y_max的距离，所以，即使使用的相同步长（Step），生成的两组数可能长度并不一样。
'''
numpy 中arange()用于生成一组连续的数，可以指定数的起点、终点、和步长。
'''
#基于生成的x值、y值，形成一个个点（二维坐标）,用于后续预测这些点的类别
xx, yy = np.meshgrid(x_value, y_value)
'''
numpy中的meshgrid()函数根据输入的两组数生成两个二维数组，并且这两个数组shape是一模一样的。
'''
#ravel()多维数据将为一维  np.c_== np.column() 按列叠加矩阵
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])#使用分类器对生成的坐标点进行预测类别
Z = Z.reshape(xx.shape)#转换预测结果Z的格式，使得可以与坐标点一一对应

#绘制原始数据的散点图（此时，类型数据是真实值） palette设置hue指定的变量的不同级别颜色。
ax = sns.scatterplot(data.Sugar, y=data.Butter, hue=label, palette='Set2')
ax.legend(loc="lower right") #图例 说明每条曲线的文字显示

#绘制基于预测值的分界面(等高线)，即分类器的边界
plt.contourf(xx, yy, Z, alpha=0.3)

plt.show()
