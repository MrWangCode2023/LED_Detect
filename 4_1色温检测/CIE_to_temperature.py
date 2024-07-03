def cie_xy_to_CCT(cie):
    x, y = cie
    n = (x - 0.3320) / (y - 0.1858)
    CCT = -449 * (n ** 3) + 3525 * (n ** 2) - 6823.3 * n + 5520.33
    return CCT


if __name__== "__main__":
    # 示例使用
    x = 0.3127  # 示例 CIE x 坐标
    y = 0.3290  # 示例 CIE y 坐标
    cct = cie_xy_to_CCT(x, y)
    print(f"计算的色温 (CCT) 是: {cct:.2f} K")
