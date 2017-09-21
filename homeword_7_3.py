# coding=utf-8
import numpy as np

# 将3.0数据集数字化,各个属性依次用1,2,3来编号
data = [[1, 2, 2, 1, 3, 1, 2, 2, 2, 1, 3, 3, 1, 3, 2, 3, 1],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 1, 1],
        [1, 2, 1, 2, 1, 1, 1, 1, 2, 3, 3, 1, 1, 2, 1, 1, 2],
        [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 3, 2, 2, 1, 3, 2],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 1, 1, 2, 3, 2],
        [1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2]]

test_data = [1, 1, 1, 1, 1, 1, 0.697, 0.460]
N_i = [3, 3, 3, 3, 3, 2]   # 不同属性的可能取值个数
N_positive = 8.0
N_negative = 9.0
N = 17

P0 = np.log((N_positive + 1) / (N + 2))  # 好瓜的类先验概率
P1 = np.log((N_negative + 1) / (N + 2))  # 坏瓜的类先验概率

p0 = [0] * 6  # 好瓜的条件概率
p1 = [0] * 6  # 坏瓜的条件概率

x = 0.0
y = 0.0

for i in range(0, 6):
    for j in range(0, 8):
        if data[i][j] == test_data[i]:
            p0[i] = p0[i] + 1

    for j in range(8, 17):
        if data[i][j] == test_data[i]:
            p1[i] = p1[i] + 1

    x += np.log((p0[i] + 1.0) / (N_positive + N_i[i]))   # 修正的拉普拉斯方式计算
    y += np.log((p1[i] + 1.0) / (N_negative + N_i[i]))

x += P0
y += P1

z = [1.959, 0.788, 1.203, 0.066]   # 密度和含糖的均值和方差

for p in z[0:2]:
    x += np.log(p)
    x += np.log(p)

for p in z[2:4]:
    y += np.log(p)

print("x=%.4f" % x)
print("y=%.4f" % y)
