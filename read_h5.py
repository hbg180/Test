import h5py
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/下载/浏览器下载/outdoor_day1_gt.hdf5'
h5f = h5py.File(path, mode='r')
# path = 'D:/下载/浏览器下载/indoor_flying4_data.bag'
# with h5py.File(path, mode='r') as h5f:
#     # for key in h5f['davis/left'].keys():
#     #     print(key, end=',')
#     print(h5f.keys())
#     data = np.array(h5f['davis/left'])
np.set_printoptions(threshold=np.inf, precision=30)
# print(data[:10])
# plt.plot()
# plt.imshow(data[0])
# plt.show()

# _gt.hdf5
# davis:left:[blended_image_rect,blended_image_rect_ts,depth_image_raw,depth_image_raw_ts,depth_image_rect,depth_image_rect_ts,flow_dist,flow_dist_ts,odometry,odometry_ts,pose,pose_ts,]

# _data.hdf5
# davis:    [left:[events,image_raw,image_raw_event_inds,image_raw_ts,imu,imu_ts],
#           right:[events,imu,imu_ts]]
# velodyne: [scans,scans_ts]
# visensor: [imu,imu_ts,left,right] 视觉传感器


print('--------------------------------------------------------')
f = np.load('D:/下载/浏览器下载/outdoor_day1_gt_flow_dist.npz')    # timestamps, y_flow_dist, x_flow_dist
# print(f['timestamps'].shape)
# print(f['y_flow_dist'][4356])
# print(f['x_flow_dist'][4356])
# (5134,)
# (5134, 260, 346)
# (5134, 260, 346)

# flow = np.load(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\optical_flow\000000.npy')
# depth = np.load(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\depth_raw\000000.npy')

left = h5py.File(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\davis\left\events\000000.h5')
right = h5py.File(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\davis\right\events\000000.h5')

# print('--------------------------------------------------------')
# f = np.load(r'E:\Git\Papers\Spike-FlowNet\datasets\indoor_flying1\count_data\0.npy')  # [2,260,346,10]
# f = np.load(r'E:\Git\Papers\Spike-FlowNet\datasets\indoor_flying1\gray_data\2600.npy')     # [260,346]
# f0 = h5py.File(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\davis\left\events\000000.h5')
# print(f.shape)
# plt.plot()
# plt.imshow(f)
# plt.show()
# print(f0['myDataset'][-1])

# f1 = h5py.File(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\davis\left\events\000001.h5')
# f2 = h5py.File(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\davis\left\events\000002.h5')
# f3 = h5py.File(r'D:\下载\浏览器下载\mvsec_outdoor_day_1_20Hz\outdoor_day_1\davis\left\events\005125.h5')
# print(f1['myDataset'][0])
print('exit!')
