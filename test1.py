import numpy as np
from dv import AedatFile
from tqdm import tqdm
# 替换为你的 .aedat4 文件路径
aedat_file_path = r'E:\tmp\dvSave-2024_03_05_19_21_32.aedat4'

# 使用 dv 库读取 .aedat4 文件
with AedatFile(aedat_file_path) as f:
    # 预先分配一个列表，用于存储事件
    events = []

    # for frame in f['frames']:
    #     pass
    # 遍历文件中的所有事件
    for event in f['events']:
        # 将每个事件的x, y, timestamp属性添加到列表中，并将polarity从布尔值转换为整数
        events.append([event.x, event.y, event.timestamp, int(event.polarity)])

    # 将事件列表转换为 NumPy 数组
    events_array = np.array(events)
np.save(r'E:\tmp\events', events_array)
# 打印结果以验证
print(events_array.shape)
print(events_array[:5])  # 打印前5个事件作为示例