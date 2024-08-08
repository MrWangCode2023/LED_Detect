import numpy as np


def t5_find_neighbors(arr, peak):
    sorted_arr = np.sort(arr)   # 降序排序
    # print("降序排序后的数组:", sorted_arr)

    left_right_x = []
    if peak in sorted_arr:
        index = np.where(sorted_arr == peak)[0][0]  # 找到目标值的索引
        left = sorted_arr[index - 1] if index > 0 else None  # 左边的值
        right = sorted_arr[index + 1] if index < len(sorted_arr) - 1 else None  # 右边的值

        left_right_x.extend([left, right])

        # 计算flare的距离，如果距离值为 None，需要处理
        distance = (right - left) if left is not None and right is not None else None
        return left_right_x, distance
    else:
        return [None, None], None  # 如果目标值不在数组中


if __name__ == "__main__":
    # 示例
    arr = [308, 310, 339, 367, 309]
    target = 339
    left_right_x, distance = t5_find_neighbors(arr, target)
    print("left:", left_right_x[0])
    print("right:", left_right_x[1])
    print("distance:", distance)
