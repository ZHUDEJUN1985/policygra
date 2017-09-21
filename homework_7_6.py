# coding=utf-8
import numpy as np

# 将西瓜3.0数据集数字化,各个属性依次用1,2,3来编号
data = [[1, 2, 2, 1, 3, 1, 2, 2, 2, 1, 3, 3, 1, 3, 2, 3, 1],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 1, 1],
        [1, 2, 1, 2, 1, 1, 1, 1, 2, 3, 3, 1, 1, 2, 1, 1, 2],
        [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 3, 2, 2, 1, 3, 2],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 1, 1, 2, 3, 2],
        [1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2]]

test_data = [1, 1, 1, 1, 1, 1, 0.697, 0.460]
N = [3, 3, 3, 3, 3, 2]      # 不同属性的可能取值个数
N_i_0 = [3, 2, 2, 2, 2, 2]  # 好瓜不同属性的可能取值个数
N_i_1 = [3, 3, 3, 3, 3, 2]  # 坏瓜不同属性的可能取值个数
N_positive = 8.0
N_negative = 9.0
D = 17.0

pc0 = [0] * 6  # 好瓜的条件概率P(c,x)
pc1 = [0] * 6  # 坏瓜的条件概率P(c,x)
pc_i_0 = np.zeros((6, 6))  # 好瓜的半朴素条件概率P(x_j|c,x_i)
pc_i_1 = np.zeros((6, 6))  # 坏瓜的半朴素条件概率P(x_j|c,x_i)

x = 0.0
y = 0.0

for i in range(0, 6):
    for k in range(0, 8):
        if data[i][k] == test_data[i]:
            pc0[i] = pc0[i] + 1

    for k in range(8, 17):
        if data[i][k] == test_data[i]:
            pc1[i] = pc1[i] + 1

    temp0 = 1.0
    temp1 = 1.0
    for j in range(0, 6):
        for r in range(0, 8):
            if data[i][r] == test_data[i] and data[j][r] == test_data[j]:
                pc_i_0[i][j] = pc_i_0[i][j] + 1

        temp0 *= (pc_i_0[i][j] + 1.0)/(pc0[i] + N_i_0[j])

        for r in range(8, 17):
            if data[i][r] == test_data[i] and data[j][r] == test_data[j]:
                pc_i_1[i][j] = pc_i_1[i][j] + 1

        temp1 *= (pc_i_1[i][j] + 1.0) / (pc1[i] + N_i_1[j])

    x += ((pc0[i] + 1.0)/(D + N[i]*N_i_0[i]))*temp0
    y += ((pc1[i] + 1.0)/(D + N[i]*N_i_1[i]))*temp1

print("x=%.4f" % x)
print("y=%.4f" % y)
