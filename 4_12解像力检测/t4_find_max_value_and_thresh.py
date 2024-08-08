import numpy as np

def t4_peak_resolution_flare(center_line, flare_ratio=0.1, flare_scale_parameter=5, resolution_ratio=0.35, resolution_scale_parameter=10):
    result = []

    if center_line.size == 0:  # 检查center_line是否为空
        return None, None, None, None, None, None



    peak = np.max(center_line)  # 找到最大值
    peak_id = np.argmax(center_line)  # 最大值的索引

    # flare value（最大值的10%）
    flare = int(peak * flare_ratio) if peak > 0 else 0
    # print(f"flare: {flare}")

    # resolution value
    reso = int(peak * resolution_ratio) if peak > 0 else 0
    # print(f"reso: {reso}")

    # 找到所有flare_indexs
    flare_ids = np.where((center_line >= flare - flare_scale_parameter) & (center_line <= flare + flare_scale_parameter))[0]

    # 找到所有reso_indexes
    reso_ids = np.where((center_line >= reso - resolution_scale_parameter) & (center_line <= reso + resolution_scale_parameter))[0]

    # 将max_index插入到thresh_indexes数组中
    flare_peak_ids = np.insert(flare_ids, 0, peak_id)  # 在开头插入max_index
    reso_peak_ids = np.insert(reso_ids, 0, peak_id)  # 在开头插入max_index

    # peak, peak_id, reso, reso_id, flare, flare_id, flare_peak_ids, reso_peak_ids
    result.extend([peak, peak_id, reso, reso_ids, flare, flare_ids, flare_peak_ids, reso_peak_ids])

    return result

# 示例使用
if __name__ == "__main__":
    center_line = np.array([10, 15, 20, 30, 25, 40, 50])
    peak_value, peak_index, flare_value, flare_indexes, flare_peak_ids, reso_peak_ids = t4_peak_resolution_flare(center_line)
    print("结果数组（包含max_index）:", flare_peak_ids)
    print("最大值索引:", flare_indexes)
