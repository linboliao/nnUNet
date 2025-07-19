import json
import math
import os
import random
import shutil

import cv2
import numpy as np
import geopandas as gpd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 100))  # 设置为1万亿像素

from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import Polygon, MultiPolygon

from wsi import WSIOperator

Image.MAX_IMAGE_PIXELS = None

LEVEL = 0


def geojson_to_mask(geojson_path, mask_path, slide_path, img_path):
    """
    将 GeoJSON 转换为二值 Mask 图像
    :param geojson_path: GeoJSON 文件路径
    :param mask_path: 输出图像路径（支持 PNG/TIFF）
    :param resolution: 输出分辨率（单位与 GeoJSON 坐标系一致）
    """
    # 1. 读取 GeoJSON 并获取边界范围
    gdf = gpd.read_file(geojson_path)

    wsi = WSIOperator(slide_path)
    width, height = wsi.level_dimensions[LEVEL]
    width0, height0 = wsi.level_dimensions[0]

    # 2. 创建全0画布
    mask = np.zeros((height0, width0), dtype=np.uint8)

    # 3. 遍历所有多边形并栅格化填充
    shapes = []
    for geom in gdf.geometry:
        # 处理 MultiPolygon（拆分为单个多边形）
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                shapes.append((poly, 1))
        elif isinstance(geom, Polygon):
            shapes.append((geom, 1))

    # 4. 栅格化：将多边形内部填充为1
    rasterized = rasterize(
        shapes,
        out_shape=(height0, width0),
        fill=0,  # 外部填充0
        all_touched=True  # 确保边界像素被覆盖
    )

    # 5. 合并到画布
    mask = np.logical_or(mask, rasterized).astype(np.uint8)
    # mask = (mask * 255).astype(np.uint8)  # 0 → 0, 1 → 255

    img = wsi.read_region((0, 0), LEVEL, (width, height))
    img.save(img_path)
    numpy_array = np.array(img)  # 形状为 (H, W, 3)

    # 3. 转换颜色通道：RGB → BGR
    img = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(mask_path, mask)
    # # 提取轮廓并绘制
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 10)  # 绿色轮廓，线宽2
    # img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
    #
    # # 用Matplotlib显示结果
    # plt.figure(figsize=(10, 6))
    # plt.imshow(img_display)
    # plt.axis('off')  # 隐藏坐标轴
    # plt.title("Image with Contours")
    # plt.show()

    print(f"Mask 已保存至: {mask_path}")


def crop_images(image_path, mask_path, output_dir, patch_size=512):
    # 创建输出目录
    image_output_dir = os.path.join(output_dir, 'imagesTr')
    mask_output_dir = os.path.join(output_dir, 'labelsTr')
    dataset = os.path.join(output_dir, 'dataset.json')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # 获取所有图片文件（支持多种格式）
    image_files = [f for f in os.listdir(image_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    count = 0
    for img_file in image_files:
        # 使用cv2读取原始图像（保留Alpha通道）
        # img = cv2.imread(os.path.join(image_path, img_file), cv2.IMREAD_UNCHANGED)
        img = Image.open(os.path.join(image_path, img_file))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if img is None:
            print(f"⚠️ Failed to load image {img_file}, skipping...")
            continue

        # 处理BGR/RGB转换（根据需求决定是否转换）
        if len(img.shape) == 3 and img.shape[2] >= 3:  # 彩色图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGB)

        height, width = img.shape[:2]

        # 处理对应mask（假设mask与原始图像同名）
        mask_file = os.path.join(mask_path, img_file)
        if not os.path.exists(mask_file):
            print(f"⚠️ Mask not found for {img_file}, skipping...")
            continue

        # 读取mask（灰度模式）
        mask = Image.open(os.path.join(mask_file))
        mask = np.array(mask)
        if mask is None:
            print(f"⚠️ Failed to load mask {mask_file}, skipping...")
            continue
        # 计算切割块数（向上取整）
        num_cols = math.ceil(width / patch_size)
        num_rows = math.ceil(height / patch_size)

        base_name = os.path.splitext(img_file)[0]

        for row in range(num_rows):
            for col in range(num_cols):
                # 计算切割坐标（处理边缘不足512px的情况）
                x1 = col * patch_size
                y1 = row * patch_size
                x2 = min(x1 + patch_size, width)
                y2 = min(y1 + patch_size, height)

                # 裁剪图像和mask
                img_patch = img[y1:y2, x1:x2]
                mask_patch = mask[y1:y2, x1:x2]

                # 填充不足512x512的部分
                if img_patch.shape[0] < patch_size or img_patch.shape[1] < patch_size:
                    # 创建填充后的图像（黑色背景）
                    padded_img = np.zeros((patch_size, patch_size, 3) if len(img.shape) == 3
                                          else (patch_size, patch_size), dtype=img.dtype)
                    padded_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)

                    # 粘贴裁剪区域
                    padded_img[:img_patch.shape[0], :img_patch.shape[1]] = img_patch
                    padded_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch

                    img_patch, mask_patch = padded_img, padded_mask

                # 检查mask是否包含有效像素（可选）
                if np.any(mask_patch > 0):
                    print(f'{img_file} {row} {col} 存在标签')
                elif random.random() > 0.75:
                    print(f'{img_file} {row} {col} 无标签，跳过')
                    continue
                # 保存子图（注意转换BGR格式）
                img_save = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR) if len(img_patch.shape) == 3 else img_patch
                cv2.imwrite(os.path.join(image_output_dir, f'MX_{count:05d}_0000.png'), img_save)
                cv2.imwrite(os.path.join(mask_output_dir, f'MX_{count:05d}.png'), mask_patch)
                count += 1

        print(f"✅ Processed {img_file} → {num_rows}x{num_cols} patches")
    data_json = {
        "channel_names": {  # formerly modalities
            "0": "R",
            "1": "G",
            "2": "B",
        },
        "labels": {  # THIS IS DIFFERENT NOW!
            "background": 0,
            "gland": 1,
        },
        "numTraining": count,
        "file_ending": ".png"
    }

    # 写入dataset.json文件
    with open(dataset, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, ensure_ascii=False, indent=4)


def lm_to_mask(label, mask_path):
    label_dict = {
        '0': 1,
        '1': 2,
        '2': 3,
        '3': 3,
        '血管': 4,
        'vessel': 4,
    }
    with open(label) as f:
        data = json.load(f)

    # 创建空白Mask（单通道）
    mask = np.zeros((data["imageHeight"], data["imageWidth"]), dtype=np.uint8)

    for shape in data["shapes"]:
        if shape["label"] not in label_dict:
            print(f'{shape["label"]} not in label_dict')
        label = label_dict.get(shape["label"], 0)
        points = np.array(shape["points"], dtype=np.int32)
        # 绘制多边形区域（填充类别ID）
        cv2.fillPoly(mask, [points], color=label)

    # 保存为PNG
    Image.fromarray(mask).save(mask_path)


if __name__ == "__main__":
    # 配置路径
    # slide_dir = '/NAS3/lbliao/Data/MXB/segment/slides'
    # geo_dir = '/NAS3/lbliao/Data/MXB/classification/第一批/label-1'
    #
    # mask_dir = '/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/mask'
    # img_dir = '/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/image'
    # patch_dir = '/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/Dataset002_GLAND'
    # os.makedirs(mask_dir, exist_ok=True)
    # os.makedirs(img_dir, exist_ok=True)
    # os.makedirs(patch_dir, exist_ok=True)
    # for slide in os.listdir(slide_dir):
    #     base, ext = os.path.splitext(slide)
    #     geo_path = os.path.join(geo_dir, f'{base}有癌.geojson')
    #     if not os.path.exists(geo_path):
    #         continue
    #     mask_path = os.path.join(mask_dir, f'{base}.png')
    #     if os.path.exists(mask_path):
    #         os.remove(mask_path)
    #     img_path = os.path.join(img_dir, f'{base}.png')
    #     if os.path.exists(img_path):
    #         os.remove(img_path)
    #     geojson_to_mask(
    #         os.path.join(geo_path),
    #         os.path.join(mask_path),
    #         os.path.join(slide_dir, slide),
    #         os.path.join(img_path),
    #     )
    # crop_images(img_dir, mask_dir, patch_dir, 1024)
    image_dir = f'/NAS3/lbliao/Data/MXB/Detection/0702/val'
    label_dir = f'/NAS3/lbliao/Data/MXB/Detection/0702/val'
    new_image_dir = f'/NAS3/lbliao/Data/MXB/Detection/0702/dataset/images'
    mask_dir = f'/NAS3/lbliao/Data/MXB/Detection/0702/dataset/masks'
    imagesTr = '/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/Dataset003_GLAND/imagesTr'
    labelsTr = '/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/Dataset003_GLAND/labelsTr'
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    # os.makedirs(new_image_dir, exist_ok=True)
    # os.makedirs(mask_dir, exist_ok=True)
    # image_extensions = (".png", ".jpg")
    # image_files = [
    #     file
    #     for file in os.listdir(image_dir)
    #     if file.lower().endswith(image_extensions)
    # ]
    # for img in image_files:
    #     img_path = os.path.join(image_dir, img)
    #     label_path = os.path.join(label_dir, img.replace('.png', '.json').replace('.jpg', '.json'))
    #     new_image_path = os.path.join(new_image_dir, img.replace('.jpg', '.png'))
    #     mask_path = os.path.join(mask_dir, img.replace('.jpg', '.png'))
    #     lm_to_mask(label_path, mask_path)
    #     shutil.copy(img_path, new_image_path)
    #     print(f'{img} finished')
    image_extensions = (".png", ".jpg")
    image_files = [
        file
        for file in os.listdir(new_image_dir)
        if file.lower().endswith(image_extensions)
    ]
    # for i, img in enumerate(tqdm(image_files)):
    #     img_file = os.path.join(new_image_dir, img)
    #     label_file = os.path.join(mask_dir, img)
    #     img = Image.open(img_file)
    #     img.resize((1024, 1024))
    #     img.save(os.path.join(imagesTr, f'MX_{i:05d}_0000.png'))
    #     label = Image.open(label_file)
    #     label.resize((1024, 1024))
    #     label.save(os.path.join(labelsTr, f'MX_{i:05d}.png'))


    def process_file(args):
        img_file, label_file, imagesTr, labelsTr, idx = args
        # 处理图像
        img = Image.open(img_file)
        img = img.resize((1024, 1024))
        img.save(os.path.join(imagesTr, f'MX_{idx:05d}_0000.png'))

        # 处理标签
        label = Image.open(label_file)
        label = label.resize((1024, 1024))
        label.save(os.path.join(labelsTr, f'MX_{idx:05d}.png'))
        return idx


    # 准备任务参数
    tasks = []
    for i, img in enumerate(image_files):
        img_file = os.path.join(new_image_dir, img)
        label_file = os.path.join(mask_dir, img)
        tasks.append((img_file, label_file, imagesTr, labelsTr, i))

    # 并行处理（根据CPU核心数调整max_workers）
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing images"):
            future.result()  # 获取结果（可处理异常）
    data_json = {
        "channel_names": {  # formerly modalities
            "0": "R",
            "1": "G",
            "2": "B",
        },
        "labels": {  # THIS IS DIFFERENT NOW!
            "background": 0,
            "prostate": 1,
            "cancer": 2,
            "hunhe": 3,
            "vessel": 4,
        },
        "numTraining": len(image_files),
        "file_ending": ".png"
    }

    # 写入dataset.json文件
    dataset = os.path.join('/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/Dataset003_GLAND/', 'dataset.json')
    with open(dataset, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, ensure_ascii=False, indent=4)
