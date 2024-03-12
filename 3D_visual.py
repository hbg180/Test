import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D


data = np.load(r'E:\Git\Papers\TMA\dsec\train\zurich_city_02_a\seq_000000.npz')
# print(data['voxel_prev'][0])
start, end = 9, 10
length = end - start
print(data['voxel_prev'][:length])
print(data['voxel_prev'][:length].shape)
print('事件率:{:.2f}%'.format(np.count_nonzero(data['voxel_prev'][start:end])/640/480/length*100))
data = h5py.File(r'E:\Git\Papers\Datasets\mvsec\indoor_flying4\indoor_flying4_data.hdf5', 'r')
# print(data['davis/left/events'][:10])
X = data['davis/left/events'][:, 0]
Y = data['davis/left/events'][:, 1]
T = data['davis/left/events'][:, 2] - data['davis/left/events'][0, 2]
P = data['davis/left/events'][:, 3]
num = 20000
color = ['r' if p == 1.0 else 'b' for p in P[:num]]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(T[:num], X[:num], Y[:num], s=1, c=color)
plt.ylim((0, 346))
ax.set_zlim((0, 260))
ax.set_xlabel('T/us')  # 画出坐标轴
ax.set_ylabel('X')
ax.set_zlabel('Y')
# plt.show()


