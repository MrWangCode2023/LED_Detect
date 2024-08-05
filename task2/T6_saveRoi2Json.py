import json

import numpy as np


def t6saveBoxes2Json(boxes, file_path):
    """
    将矩形框坐标保存到 JSON 文件。

    参数：
    - boxes: 矩形框坐标列表，每个坐标是一个包含四个顶点的列表
    - file_path: 文件保存路径
    """
    # 将 boxes 中的 NumPy 数组转换为列表
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    elif isinstance(boxes, list):
        boxes = [item.tolist() if isinstance(item, np.ndarray) else item for item in boxes]

    with open(file_path, 'w') as f:
        json.dump(boxes, f, indent=4)


if __name__ == "__main__":
    # 示例数据
    boxes = [
        [[2, 533], [15, 522], [26, 534], [13, 546]],
        [[15, 521], [28, 510], [39, 522], [27, 534]],
        # ... 其他矩形框
    ]

    t6saveBoxes2Json(boxes, 'boxes.json')
