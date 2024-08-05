import numpy as np

# 示例 framesDatas 数据
framesDatas = [
    [0, [255, 0, 0], [0, 255, 255], 1],  # 结构示例
    [1, [0, 255, 0], [120, 255, 255], 0],
    # ...
]

# 检查 framesDatas 的一致性
for i, data in enumerate(framesDatas):
    print(f"Item {i} shape: {len(data)}")

# 确保数据一致性，假设每个元素应该有 4 个值
if all(len(data) == 4 for data in framesDatas):
    results = np.array(framesDatas)
else:
    print("Data has inconsistent lengths, converting to object array.")
    results = np.array(framesDatas, dtype=object)

print(results)
