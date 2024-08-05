import json


def t7LoadRoisFromJson(roisPath):
    """
    从 JSON 文件加载矩形框坐标。

    参数：
    - file_path: 文件路径

    返回：
    - boxes: 矩形框坐标列表
    """
    with open(roisPath, 'r') as f:
        boxes = json.load(f)
        roi_nums = len(boxes)

    return boxes, roi_nums


if __name__ == "__main__":
    # 读取数据
    boxes = t7LoadRoisFromJson('roisPath.json')
    print(boxes)
