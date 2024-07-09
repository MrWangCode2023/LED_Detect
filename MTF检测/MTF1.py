from Box import bbox, mbbox
from ESF import ESF
from ESF_to_LSF import ESF_to_LSF
from LSF_to_MTF import LSF_to_MTF


def MTF(image):
    line1 = bbox(image)
    line2 = mbbox(image)

    # 计算ESF
    esf = ESF(line1)
    # 求导微分
    lsf = ESF_to_LSF(esf)
    # 一维傅里叶变换
    mtf = LSF_to_MTF(lsf)
    return mtf


