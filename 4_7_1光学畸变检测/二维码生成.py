# import qrcode
#
# # Define the data from the user
# data = """
# 危险废物
# 废物名称: 废切削液
# 危险特性: 毒性
# 废物类别: 废乳化液
# 废物代码: 900-006-09
# 废物形态: L液态
# 主要成分: 油、乳化剂和添加剂
# 有害成分: 亚硝酸钠
# 注意事项:
# 数字识别码:
# 产生/收集单位: 武汉正光
# 联系人: 程钢
# 联系方式: 02769372289
# """
#
# # Create QR code instance
# qr = qrcode.QRCode(
#     version=1,  # controls the size of the QR Code, 1 is the smallest version
#     error_correction=qrcode.constants.ERROR_CORRECT_L,  # controls the error correction used for the QR Code
#     box_size=10,  # controls how many pixels each “box” of the QR code is
#     border=4,  # controls how many boxes thick the border should be
# )
#
# # Add data to the instance
# qr.add_data(data)
# qr.make(fit=True)
#
# # Create an image from the QR Code instance
# img = qr.make_image(fill_color="black", back_color="white")
#
# # Save the image
# img.save("qr_code_waste.png")
#
# # Display the image
# img.show()
import qrcode

# 数据
data = """
危险废物
废物名称: 废切削液
危险特性: 毒性
废物类别: 废乳化液
废物代码: 900-006-09
废物形态: L液态
主要成分: 油、乳化剂和添加剂
有害成分: 亚硝酸钠
注意事项: 注意防火、防漏
数字识别码: 91420114303563353A 900-006-09 20240701 001
产生/收集单位: 武汉正光
联系人: 程钢
联系方式: 02769372289
"""

# 创建二维码实例
qr = qrcode.QRCode(
    version=1,  # 控制二维码的大小，值越大二维码越大 (1到40)
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # 控制二维码的错误纠正级别
    box_size=10,  # 控制二维码中每个格子的大小
    border=4,  # 控制二维码边框的大小
)

# 添加数据到二维码实例
qr.add_data(data)
qr.make(fit=True)

# 创建二维码图像
img = qr.make_image(fill_color="black", back_color="white")

# 保存图像
img.save("qr_code_full.png")

# 显示图像
img.show()

