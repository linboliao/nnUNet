import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def Visualization(mask_path, img_path):
    # 1. 读取mask图像（确保是单通道二值图）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 强制转为灰度图

    # 2. 二值化处理（确保像素值为0或255）
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # 3. 查找轮廓
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,  # 只检测外层轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
    )

    # 4. 创建可视化画布
    img = cv2.imread(img_path)
    cv2.drawContours(
        img,
        contours,
        -1,  # -1表示绘制所有轮廓
        (0, 0, 255),  # BGR颜色：红色
        2  # 轮廓线粗细
    )
    cv2.imwrite(mask_path.replace('prediction_results', 'prediction_results2'), img)

    # # 6. 正确显示图像（BGR转RGB）[1](@ref)
    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换颜色空间
    # plt.title("轮廓可视化结果")
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    mask_dir = f'/NAS3/lbliao/Data/MXB/seminal/prediction_results'
    img_dir = f'/NAS3/lbliao/Data/MXB/seminal/wsi_patches'
    masks = os.listdir(mask_dir)
    for mask in masks:
        if mask.endswith(".png"):
            mask_path = os.path.join(mask_dir, mask)
            img_path = os.path.join(img_dir, mask.replace(".png", "_0000.png"))
            Visualization(mask_path, img_path)
