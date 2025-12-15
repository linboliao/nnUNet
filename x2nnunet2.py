import json
import math
import os
import random

import cv2
import numpy as np
import geopandas as gpd
from tqdm import tqdm

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 100))  # 设置为1万亿像素

from PIL import Image
from rasterio.features import rasterize
from pathlib import Path
from wsi import WSIOperator
from concurrent.futures import ProcessPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None

LEVEL = 0
patch_size = 1024
name_to_id = {}
id_counter = 1

def geojson_to_mask(geojson_path, slide_path, output_dir):
    """
    将 GeoJSON 转换为二值 Mask 图像
    :param geojson_path: GeoJSON 文件路径
    :param mask_path: 输出图像路径（支持 PNG/TIFF）
    :param resolution: 输出分辨率（单位与 GeoJSON 坐标系一致）
    """
    # 1. 读取 GeoJSON 并获取边界范围
    global id_counter
    gdf = gpd.read_file(geojson_path)

    wsi = WSIOperator(slide_path)
    width, height = wsi.level_dimensions[LEVEL]
    width0, height0 = wsi.level_dimensions[0]

    # 3. 遍历所有多边形并栅格化填充
    shapes = []
    unknow = []
    # 遍历GeoDataFrame的每一行（同时获取几何体和属性）


    for idx, row in gdf.iterrows():
        geom = row.geometry
        # 根据分类名称设置填充值

        # classification = row['classification']
        # if isinstance(classification, str):
        #     classification = json.loads(classification)
        # if not classification:
        #     fill_value = 2
        # elif classification['name'] in ['prostate', 'Negative', 'non-cancer']:
        #     fill_value = 1
        # elif classification['name'] in ['cancer', 'Positive']:
        #     fill_value = 2
        # elif classification['name'] == 'Other':
        #     continue
        # elif classification['name'] in ['lymphocyte', 'vessle', 'nerve', '杂质']:
        #     fill_value = 3
        # else:
        #     fill_value = 2
        name = row['name']
        if name not in name_to_id:
            name_to_id[name] = id_counter
            id_counter += 1
        fill_value = name_to_id[name]

        # 处理MultiPolygon（拆分为单个多边形）
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                shapes.append((poly, fill_value))
        # 处理Polygon
        elif geom.geom_type == 'Polygon':
            shapes.append((geom, fill_value))
    print(set(unknow))
    # 栅格化：根据分类名称填充不同值
    rasterized = rasterize(
        shapes,
        out_shape=(height0, width0),
        fill=0,  # 背景填充0
        all_touched=True  # 确保边界像素被覆盖
    )

    mask = rasterized

    if LEVEL != 0:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

    slide = Path(slide_path).stem
    print(f"开始切 patch {slide}")
    image_output_dir = os.path.join(output_dir, 'imagesTr')
    mask_output_dir = os.path.join(output_dir, 'labelsTr')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    count = len(os.listdir(image_output_dir))
    # 计算切割块数（向上取整）
    num_cols = math.ceil(width / patch_size)
    num_rows = math.ceil(height / patch_size)
    print(f'列：{num_cols} 行： {num_rows}')
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算切割坐标（处理边缘不足512px的情况）
            x1 = col * patch_size
            y1 = row * patch_size
            x2 = min(x1 + patch_size, width)
            y2 = min(y1 + patch_size, height)

            # 裁剪图像和mask
            img_patch = wsi.read_region((x1, y1), LEVEL, (min(patch_size, width - x1), min(patch_size, height - y1)))
            img_patch = img_patch.convert('RGB')
            mask_patch = mask[y1:y2, x1:x2]
            # if np.isin(2, mask_patch).any():
            #     print(f'{slide} 行 {row} 列 {col} 存在癌症标签')
            # elif np.isin(1, img_patch).any():
            #     print(f'{slide} 行 {row} 列 {col} 存在腺体标签')

            if img_patch.width < patch_size or img_patch.height < patch_size:
                # 创建填充后的图像（黑色背景）
                # 将 img_patch 不足 patch size 部分用0 填充，img_patch是 PIL image
                padded_img = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))

                padded_img.paste(img_patch)
                padded_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)
                padded_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch

                img_patch, mask_patch = padded_img, padded_mask

            # # 检查mask是否包含有效像素（可选）
            # if np.any(mask_patch > 0):
            #     print(f'{slide} {row} {col} 存在标签')
            # elif random.random() < 0.7:
            #     print(f'{slide} {row} {col} 无标签，跳过')
            #     continue
            # else:
            #     print(f'{slide} {row} {col} 无标签，保存')
            # 保存子图（注意转换BGR格式）
            # img_patch = img_patch.resize((1024, 1024))
            img_patch.save(os.path.join(image_output_dir, f'MX_{count:05d}_0000.png'))
            mask_patch = np.resize(mask_patch, (1024, 1024))
            cv2.imwrite(os.path.join(mask_output_dir, f'MX_{count:05d}.png'), mask_patch)
            count += 1
    print(f"共保存图片{count}张")

    print(f"✅ Processed {slide} → {num_rows}x{num_cols} patches")
    return name_to_id


if __name__ == "__main__":
    slide_dir = '/NAS3/lbliao/Data/MXB/segment/0716/slides'
    geo_dir = '/NAS3/lbliao/Data/MXB/segment/0716/manual'
    patch_dir = '/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/Dataset001'
    slides = [
        '/NAS145/Data/JF-cells/slides',
        # '/NAS3/lbliao/Data/MXB/segment/0716/slides',
        # '/NAS3/lbliao/Data/MXB/segment/YNZL修订/slides',
        # '/NAS3/lbliao/Data/MXB/segment/无癌病例/slides',
        # '/NAS3/lbliao/Data/MXB/segment/补充的分割图像/slides',
        # '/NAS3/lbliao/Data/MXB/segment/补充神经节标注/slides',
    ]
    geos = [
        '/NAS145/Data/JF-cells/merger',
        # '/NAS3/lbliao/Data/MXB/segment/0716/manual',
        # '/NAS3/lbliao/Data/MXB/segment/YNZL修订/geojson',
        # '/NAS3/lbliao/Data/MXB/segment/无癌病例/geojson',
        # '/NAS3/lbliao/Data/MXB/segment/补充的分割图像/geojson',
        # '/NAS3/lbliao/Data/MXB/segment/补充神经节标注/geojson',
    ]
    for slide_dir, geo_dir in zip(slides, geos):
        tasks = []
        slide_paths = []
        geo_paths = []
        for slide in os.listdir(slide_dir):
            base, ext = os.path.splitext(slide)
            geo_path = os.path.join(geo_dir, f'{base}.geojson')
            if not os.path.exists(geo_path):
                print(f'{slide} 没有标签！！！')
                continue
            slide_path = os.path.join(slide_dir, slide)
            geojson_to_mask(geo_path, slide_path, patch_dir)
    # 创建数据字典（保持原始结构）
    data = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        # "labels": {"background": 0, "prostate": 1, "cancer": 2, "other": 3},
        "labels": name_to_id,
        "numTraining": len(os.listdir(os.path.join(patch_dir, 'imagesTr'))),
        "file_ending": ".png"
    }

    # 使用 Pathlib 安全构建路径（跨平台兼容）
    data_path = Path(patch_dir) / "data.json"

    # 将数据写入 JSON 文件（带格式化和错误处理）
    try:
        with open(data_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)  # 添加缩进增强可读性
        print(f"数据成功写入: {data_path}")
    except (IOError, OSError) as e:
        print(f"文件写入失败: {str(e)}")
    except TypeError as e:
        print(f"数据类型错误: {str(e)}")
