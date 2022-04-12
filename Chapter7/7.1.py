import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
from mpl_toolkits import mplot3d # 三维画图
import numpy as np

def f(x):
    return x * np.cos(np.pi * x)


#局部极小值与全局极小值
d2l.set_figsize((4.5, 2.5))
x = np.arange(-1.0, 2.0, 0.1)
fig, = d2l.plt.plot(x, f(x))
#annotate用于在图形上给数据添加文本注解，而且支持带箭头的划线工具，方便我们在合适的位置添加描述信息。
fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77,-1.0), arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8), arrowprops=dict(arrowstyle='->'))
d2l.plt.xlabel('x')
d2l.plt.ylabel('f(x)')
d2l.plt.show()


#鞍点值
x = np.arange(-2.0, 2.0, 0.1)
fig, = d2l.plt.plot(x, x**3)
fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52,-5.0), arrowprops=dict(arrowstyle='->'))
d2l.plt.xlabel('x')
d2l.plt.ylabel('f(x)')
d2l.plt.show()
#鞍点例二
x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]#步长为复数表示点数
z = x**2 - y**2
#“111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})#网格绘制
ax.plot([0], [0], [0], 'rx')#标记点
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.show()