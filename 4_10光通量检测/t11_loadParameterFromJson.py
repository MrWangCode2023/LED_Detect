import json
import numpy as np
import os


def t11_loadParameterFromJson(filename):
    """
    从 JSON 文件读取映射矩阵和多项式系数。

    Args:
        filename: JSON 文件名

    Returns:
        T, T_inv, coefficients, p0, p1: 从文件中读取的值
    """
    # 检查文件是否存在
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"文件 {filename} 不存在。")

    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)

        T = np.array(data['T'])  # 将列表转换为 NumPy 数组
        T_inv = np.array(data['T_inv'])
        coefficients = np.array(data['coefficients'])

        # 可选参数 p0 和 p1 的处理
        p0 = np.array(data['p0']) if 'p0' in data else None
        p1 = np.array(data['p1']) if 'p1' in data else None

        return T, T_inv, coefficients, p0, p1

    except json.JSONDecodeError:
        raise ValueError("JSON 文件格式不正确。")
    except KeyError as e:
        raise KeyError(f"缺少必要的键: {e}")
    except Exception as e:
        raise Exception(f"读取数据时发生错误: {e}")


# 示例用法
if __name__ == "__main__":
    try:
        T, T_inv, coefficients, p0, p1 = t11_loadParameterFromJson("parameters.json")
        print("T:", T)
        print("T_inv:", T_inv)
        print("coefficients:", coefficients)
        if p0 is not None:
            print("p0:", p0)
        if p1 is not None:
            print("p1:", p1)
    except Exception as e:
        print(e)
