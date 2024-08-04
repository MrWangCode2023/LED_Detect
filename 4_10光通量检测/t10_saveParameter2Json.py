import json
import numpy as np

def t10_saveParameter2Json(filename, T, T_inv, coefficients, p0=None, p1=None):
    """
    将映射矩阵和多项式系数保存到 JSON 文件。

    Args:
        filename: JSON 文件名
        T: 坐标系映射矩阵 (NumPy 数组)
        T_inv: 逆映射矩阵 (NumPy 数组)
        coefficients: 拟合的多项式系数 (NumPy 数组)
        p0: 可选，表示映射关系的起始点，默认为 None
        p1: 可选，表示映射关系的结束点，默认为 None
    """
    # 类型检查
    if not isinstance(T, np.ndarray) or not isinstance(T_inv, np.ndarray) or not isinstance(coefficients, np.ndarray):
        raise ValueError("T, T_inv, and coefficients must be NumPy arrays.")

    data = {
        'T': T.tolist(),           # 将 NumPy 数组转换为列表
        'T_inv': T_inv.tolist(),
        'coefficients': coefficients.tolist()
    }

    # 添加 p0 和 p1 到 JSON 数据（如果提供的话）
    if p0 is not None:
        data['p0'] = p0.tolist() if isinstance(p0, np.ndarray) else p0
    if p1 is not None:
        data['p1'] = p1.tolist() if isinstance(p1, np.ndarray) else p1

    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"数据已保存到 {filename}")
    except Exception as e:
        print(f"保存数据时发生错误: {e}")

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    T = np.array([[1, 0], [0, 1]])
    T_inv = np.array([[1, 0], [0, 1]])
    coefficients = np.array([0.1, 0.2, 0.3])
    p0 = np.array([0, 0])
    p1 = np.array([1, 1])

    # 保存到 JSON 文件
    t10_saveParameter2Json("parameters.json", T, T_inv, coefficients, p0, p1)
