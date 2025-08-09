import os
import argparse
import numpy as np
from PIL import Image
import cv2
import geojson
from multiprocessing import Pool
import glob
import tqdm
import json
import sys
import subprocess
import shutil
from pathlib import Path
from wsi import WSIOperator

i = 0
# 配置环境变量
os.environ["PYTHONPATH"] = "/NAS3/lbliao/Code/nnUNet:$PYTHONPATH"
os.environ["nnUNet_raw"] = "/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/"
os.environ["nnUNet_preprocessed"] = "/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_trained_models"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{i}"

# 固定参数
LEVEL = 0
PATCH_SIZE = 1024
NNUNET_DATASET_ID = f"Dataset005_GLAND"
CONFIG_NAME = "2d"
TEMP_IMAGE_DIR = f"/NAS3/lbliao/Data/MXB/Detection/0318/tmp/wsi_patches_{i}"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
PREDICTION_OUTPUT_DIR = f"/NAS3/lbliao/Data/MXB/Detection/0318/tmp/prediction_results_{i}"
COORD_MAPPING_FILE = os.path.join(TEMP_IMAGE_DIR, "coord_mapping.json")  # 坐标映射文件


def init_worker(wsi_path):
    global _worker_slide
    _worker_slide = WSIOperator(wsi_path)


def process_patch(args):
    index, coord, output_dir = args
    try:
        x, y = coord
        patch = _worker_slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE))
        patch_rgb = patch.convert("RGB")
        # 使用索引格式保存文件名
        patch_path = os.path.join(output_dir, f"MX_{index:05d}_0000.png")
        patch_rgb.save(patch_path)
        return index, coord
    except Exception as e:
        print(f"Error processing patch at ({x},{y}): {str(e)}")
        return index, None


def extract_contours(mask_image):
    """从掩码图像中提取轮廓并转换为GeoJSON格式
    值1: prostate (绿色)
    值2: cancer (红色)
    """
    mask_np = np.array(mask_image, dtype=np.uint8)

    # 创建不同类别的二值化掩码
    masks = [np.where(mask_np == i, 255, 0).astype(np.uint8) for i in range(1, np.max(mask_np)+1)]
    pp_dict = {
        1: {"classification": {"name": "prostate", "color": [0, 255, 0]}},
        2: {"classification": {"name": "cancer", "color": [255, 0, 0]}},
    }
    features = []
    for i, mask in enumerate(masks):

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            coordinates = []
            for point in approx:
                pt = point[0]
                coordinates.append([float(pt[0]), float(pt[1])])

            if not np.array_equal(coordinates[0], coordinates[-1]):
                coordinates.append(coordinates[0])

            poly = geojson.Polygon([coordinates])
            feature = geojson.Feature(
                geometry=poly,
                properties=pp_dict[i + 1]
            )
            features.append(feature)
            if i == 2:
                print(f'存在 癌症标签')

    return geojson.FeatureCollection(features)


# def extract_contours(mask_image):
#     """从掩码图像中提取轮廓并转换为GeoJSON格式"""
#     mask_np = np.array(mask_image)
#     contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     features = []
#     for contour in contours:
#         # 多边形近似平滑
#         epsilon = 0.001 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#
#         # 转换为geojson格式 (注意坐标转换: [x,y] -> [列,行])
#         coordinates = []
#         for point in approx:
#             pt = point[0]  # 轮廓点的格式为[[x,y]]
#             coordinates.append([float(pt[0]), float(pt[1])])
#
#         # 确保多边形闭合
#         if not np.array_equal(coordinates[0], coordinates[-1]):
#             coordinates.append(coordinates[0])
#
#         poly = geojson.Polygon([coordinates])
#         features.append(geojson.Feature(geometry=poly))
#
#     return geojson.FeatureCollection(features)


def main(wsi_path, output_geojson):
    # 清空临时目录
    shutil.rmtree(TEMP_IMAGE_DIR, ignore_errors=True)
    shutil.rmtree(PREDICTION_OUTPUT_DIR, ignore_errors=True)
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    # 1. 处理WSI文件
    try:
        slide = WSIOperator(wsi_path)
        level0_dim = slide.dimensions
        level2_dim = slide.level_dimensions[LEVEL]

        # 计算下采样因子
        downsample_x = level0_dim[0] / level2_dim[0]
        downsample_y = level0_dim[1] / level2_dim[1]
        print(f"WSI信息: Level0尺寸={level0_dim}, Level{LEVEL}尺寸={level2_dim}, 下采样因子=({downsample_x:.2f}, {downsample_y:.2f})")

        # 2. 生成patch坐标
        coords = []
        for y in range(0, level2_dim[1], PATCH_SIZE):
            for x in range(0, level2_dim[0], PATCH_SIZE):
                coords.append((int(x * downsample_x), int(y * downsample_y)))

        # 3. 并行处理patch提取
        # 使用索引命名patch文件
        print(f"正在从WSI提取{len(coords)}个patch...")
        pool_args = [(idx, coord, TEMP_IMAGE_DIR) for idx, coord in enumerate(coords)]

        coord_mapping = {}
        with Pool(
                processes=os.cpu_count() // 2,
                initializer=init_worker,
                initargs=(wsi_path,)
        ) as pool:
            results = list(tqdm.tqdm(pool.imap(process_patch, pool_args), total=len(coords)))

        # 收集坐标映射信息
        for index, coord in results:
            if coord is not None:
                coord_mapping[index] = coord

        # 保存坐标映射到JSON文件
        with open(COORD_MAPPING_FILE, "w") as f:
            json.dump(coord_mapping, f)
        print(f"坐标映射已保存至: {COORD_MAPPING_FILE}")
        print(f"成功提取{len(coord_mapping)}/{len(coords)}个patch至目录: {TEMP_IMAGE_DIR}")

    except Exception as e:
        print(f"WSI处理错误: {str(e)}")
        return

    # 4. 执行nnUNet预测
    try:
        print("执行nnUNet预测...")
        predict_cmd = [
            "nnUNetv2_predict",
            "-i", TEMP_IMAGE_DIR,
            "-o", PREDICTION_OUTPUT_DIR,
            "-d", NNUNET_DATASET_ID,
            "-c", CONFIG_NAME
        ]
        subprocess.run(predict_cmd, check=True)
        print(f"预测结果已保存至: {PREDICTION_OUTPUT_DIR}")
    except Exception as e:
        print(f"预测执行错误: {str(e)}")
        return

    # 5. 转换并合并预测结果
    try:
        print("处理预测结果并生成GeoJSON...")

        # 加载坐标映射
        with open(COORD_MAPPING_FILE, "r") as f:
            coord_mapping = json.load(f)
            # 将字符串键转换为整数
            coord_mapping = {int(k): v for k, v in coord_mapping.items()}

        feature_collections = []
        prediction_files = glob.glob(os.path.join(PREDICTION_OUTPUT_DIR, "*.png"))

        for pred_file in prediction_files:
            # 从文件名解析索引
            filename = os.path.basename(pred_file)
            if not filename.startswith("MX_"):
                continue
            try:
                index = int(filename[:-4].split("_")[1])
            except (IndexError, ValueError):
                continue

            # 获取该patch对应的原始坐标
            if index not in coord_mapping:
                continue
            orig_x, orig_y = coord_mapping[index]

            # 转换分割结果
            pred_image = Image.open(pred_file).convert("L")
            contours = extract_contours(pred_image)

            # 修正坐标至原始位置
            for feature in contours["features"]:
                valid_rings = []  # 存储有效环的列表

                for ring in feature.geometry.coordinates:
                    # 修正坐标
                    for point in ring:
                        point[0] = point[0] * downsample_x + orig_x
                        point[1] = point[1] * downsample_y + orig_y

                    # 保留长度大于3的环
                    if len(ring) > 3:
                        valid_rings.append(ring)

                # 如果当前feature存在有效环，则更新其geometry并加入feature_collections
                if valid_rings:
                    # 更新feature的坐标（替换为有效环）
                    feature.geometry.coordinates = valid_rings
                    feature_collections.append(feature)

        # 6. 保存GeoJSON
        output_geojson_path = output_geojson
        with open(output_geojson_path, "w") as f:
            geojson.dump(geojson.FeatureCollection(feature_collections), f)

        print(f"轮廓已保存至: {output_geojson_path}")

    except Exception as e:
        print(f"轮廓生成错误: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI分割与轮廓生成工具")
    parser.add_argument("--wsi_folder", default='/NAS3/lbliao/Data/MXB/gleason/ynzl/slides', help="输入WSI文件路径 (.svs/.tif/.ndpi)")
    parser.add_argument("--output_folder", default='/NAS3/Data1/lbliao/Data/MXB/gleason/ynzl/nnunet', help="输出GeoJSON文件路径")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    wsis = os.listdir(args.wsi_folder)
    step = 100
    for wsi in wsis[step * i: step * (i + 1)]:
        if wsi not in ['20250244312.svs','2025099815.svs']:
            continue
        wsi_path = os.path.join(args.wsi_folder, wsi)
        basename = Path(wsi_path).stem
        output_path = os.path.join(args.output_folder, f"{basename}.geojson")
        main(wsi_path, output_path)
