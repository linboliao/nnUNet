import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
import geojson
import multiprocessing
from multiprocessing.pool import Pool
import glob
import tqdm
import json
import shutil
from pathlib import Path
import tempfile
from multiprocessing import get_context

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from wsi import WSIOperator


class NoDaemonProcess(get_context("fork").Process):
    """自定义非守护进程类，解除守护进程不能创建子进程的限制"""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonPool(Pool):
    """自定义进程池，内部使用非守护进程"""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = get_context("fork")
        super().__init__(*args, **kwargs)

    def Process(self, *args, **kwds):
        proc = super().Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess  # 重写进程类型
        return proc


# 配置环境变量
os.environ["PYTHONPATH"] = "/NAS3/lbliao/Code/nnUNet:$PYTHONPATH"
os.environ["nnUNet_raw"] = "/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/"
os.environ["nnUNet_preprocessed"] = "/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_trained_models"
NNUNET_DATASET_ID = "Dataset005_GLAND"
CONFIG_NAME = "2d"
LEVEL = 0
PATCH_SIZE = 1024

# 全局初始化变量（避免冲突）
_worker_slide = None


def init_worker(wsi_path):
    global _worker_slide
    _worker_slide = WSIOperator(wsi_path)


def process_patch(args):
    index, coord, output_dir = args
    try:
        x, y = coord
        patch = _worker_slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE))
        patch_rgb = patch.convert("RGB")
        patch_path = os.path.join(output_dir, f"MX_{index:05d}_0000.png")
        patch_rgb.save(patch_path)
        return index, coord
    except Exception as e:
        print(f"Error processing patch at ({x},{y}): {str(e)}")
        return index, None


def extract_contours(mask_image):
    mask_np = np.array(mask_image, dtype=np.uint8)
    masks = [np.where(mask_np == i, 255, 0).astype(np.uint8) for i in range(1, np.max(mask_np) + 1)]
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
    return geojson.FeatureCollection(features)


def process_one_wsi(args):
    """处理单个WSI的完整流程（包含临时目录隔离）"""
    wsi_path, output_folder, gpu_index, temp_base_dir = args

    # 为当前进程创建唯一临时目录
    wsi_name = Path(wsi_path).stem
    temp_dir = tempfile.mkdtemp(prefix=f"wsi_{wsi_name}_", dir=temp_base_dir)
    temp_image_dir = os.path.join(temp_dir, "wsi_patches")
    prediction_output_dir = os.path.join(temp_dir, "prediction_results")
    coord_mapping_file = os.path.join(temp_dir, "coord_mapping.json")
    os.makedirs(temp_image_dir, exist_ok=True)
    os.makedirs(prediction_output_dir, exist_ok=True)

    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    torch.cuda.set_device(int(gpu_index))

    try:
        # 1. 处理WSI文件
        slide = WSIOperator(wsi_path)
        level0_dim = slide.dimensions
        level2_dim = slide.level_dimensions[LEVEL]
        downsample_x = level0_dim[0] / level2_dim[0]
        downsample_y = level0_dim[1] / level2_dim[1]

        # 2. 生成patch坐标
        coords = []
        for y in range(0, level2_dim[1], PATCH_SIZE):
            for x in range(0, level2_dim[0], PATCH_SIZE):
                coords.append((int(x * downsample_x), int(y * downsample_y)))

        # 3. 并行提取patch（使用非守护进程池）
        coord_mapping = {}
        pool_args = [(idx, coord, temp_image_dir) for idx, coord in enumerate(coords)]

        # 关键修复：使用标准Pool而非守护进程池
        with Pool(processes=multiprocessing.cpu_count() // 2,
                  initializer=init_worker,
                  initargs=(wsi_path,)) as pool:
            results = list(tqdm.tqdm(pool.imap(process_patch, pool_args), total=len(coords)))

        for index, coord in results:
            if coord is not None:
                coord_mapping[index] = coord

        with open(coord_mapping_file, "w") as f:
            json.dump(coord_mapping, f)

        # 4. nnUNet预测
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=False
        )
        predictor.initialize_from_trained_model_folder(
            os.path.join(os.environ["nnUNet_results"], f'{NNUNET_DATASET_ID}/nnUNetTrainer__nnUNetPlans__{CONFIG_NAME}'),
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name='checkpoint_final.pth',
        )
        predictor.predict_from_files(
            temp_image_dir,
            prediction_output_dir,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=4
        )

        # 5. 合并预测结果
        feature_collections = []
        prediction_files = glob.glob(os.path.join(prediction_output_dir, "*.png"))

        for pred_file in prediction_files:
            filename = os.path.basename(pred_file)
            if not filename.startswith("MX_"): continue
            try:
                index = int(filename[:-4].split("_")[1])
            except:
                continue
            if index not in coord_mapping: continue

            orig_x, orig_y = coord_mapping[index]
            pred_image = Image.open(pred_file).convert("L")
            contours = extract_contours(pred_image)

            for feature in contours["features"]:
                valid_rings = []
                for ring in feature.geometry.coordinates:
                    for point in ring:
                        point[0] = point[0] * downsample_x + orig_x
                        point[1] = point[1] * downsample_y + orig_y
                    if len(ring) > 3:
                        valid_rings.append(ring)
                if valid_rings:
                    feature.geometry.coordinates = valid_rings
                    feature_collections.append(feature)

        # 6. 保存结果
        output_geojson = os.path.join(output_folder, f"{wsi_name}.geojson")
        with open(output_geojson, "w") as f:
            geojson.dump(geojson.FeatureCollection(feature_collections), f)

        print(f"[{wsi_name}] 处理完成 → {output_geojson}")

    except Exception as e:
        print(f"[{wsi_name}] 处理失败: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)  # 清理临时文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多WSI并行分割与轮廓生成")
    parser.add_argument("--wsi_folder", default='/NAS3/lbliao/Data/MXB/gleason/ynzl/slides', help="WSI文件夹路径")
    parser.add_argument("--output_folder", default='/NAS3/lbliao/Data/MXB/gleason/ynzl/nnunet', help="输出GeoJSON文件夹")
    parser.add_argument("--gpus", default="0", help="可用GPU列表，如'0,1,2'")
    parser.add_argument("--temp_dir", default="/tmp", help="临时文件基础目录")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # 获取WSI文件列表
    wsi_paths = []
    for ext in ["*.svs", "*.tif", "*.ndpi"]:
        wsi_paths.extend(glob.glob(os.path.join(args.wsi_folder, ext)))

    if not wsi_paths:
        print("未找到WSI文件!")
        sys.exit(1)

    # 准备GPU轮询分配
    gpu_list = [gpu.strip() for gpu in args.gpus.split(",")]
    gpu_cycle = (gpu_list * (len(wsi_paths) // len(gpu_list) + 1))[:len(wsi_paths)]

    # 准备任务参数
    tasks = [
        (wsi_path, args.output_folder, gpu_cycle[i], args.temp_dir)
        for i, wsi_path in enumerate(wsi_paths)
    ]

    # 关键修复：使用自定义非守护进程池
    worker_count = args.workers or min(len(wsi_paths), len(gpu_list), multiprocessing.cpu_count())
    print(f"启动 {worker_count} 个进程处理 {len(wsi_paths)} 个WSI, GPU分配: {gpu_cycle}")

    # 使用自定义进程池（解决守护进程嵌套问题）
    with NoDaemonPool(processes=worker_count) as pool:
        results = list(tqdm.tqdm(pool.imap(process_one_wsi, tasks), total=len(tasks)))

    print("所有WSI处理完成!")
