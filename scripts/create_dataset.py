#!/usr/bin/env python3
"""
YOLO 训练集创建脚本

从标注数据源目录读取图片和标注，按可配置比例拆分为
训练集/验证集/测试集，输出 YOLO 格式的数据集。

用法:
    # 单个场景目录
    python scripts/create_dataset.py -s Root/03_label_check/scene_0001 -n 0002_20270812

    # 多个场景目录
    python scripts/create_dataset.py -s Root/03_label_check/scene_0001 Root/03_label_check/scene_0002 -n 0002_20270812

    # 任意文件夹（递归遍历所有图片和标注）
    python scripts/create_dataset.py -s /abc -n my_dataset

    # 自定义参数
    python scripts/create_dataset.py -s Root/03_label_check -n 0003_20270901 --split 8 1 1
    python scripts/create_dataset.py -s Root/03_label_check -n test_run --dry-run
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从标注数据创建 YOLO 训练数据集"
    )
    script_dir = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--source", "-s",
        required=True,
        nargs="+",
        help="标注数据源目录，可指定多个。支持: (1) 03_label_check 下的场景目录 "
             "(2) 任意文件夹，递归遍历其中所有图片和标注",
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="数据集名称",
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        default=str(script_dir / "Root" / "04_dataset"),
        help="数据集输出根目录 (默认: %(default)s)",
    )
    parser.add_argument(
        "--classes", "-c",
        default=str(script_dir / "Root" / "00_config" / "classes.txt"),
        help="类别文件路径 (默认: %(default)s)",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=int,
        default=[7, 2, 1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="训练/验证/测试 比例 (默认: 7 2 1)",
    )
    parser.add_argument(
        "--pos-neg-ratio",
        nargs=2,
        type=int,
        default=[9, 1],
        metavar=("POS", "NEG"),
        help="正样本/空样本 比例 (默认: 9 1)",
    )
    parser.add_argument(
        "--rounding",
        choices=["floor", "nearest"],
        default="floor",
        help="空样本数量取整方式: floor=向下取整(严格), nearest=四舍五入 (默认: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览分配方案，不写入文件 (默认: 关闭)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出目录 (默认: 关闭)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 类别加载
# ---------------------------------------------------------------------------

def load_classes(path: str) -> list:
    """
    从 classes.txt 加载类别名称列表。
    行号 = class_id，跳过 # 开头的注释行和空行。
    返回: [name_0, name_1, ...]
    """
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


# ---------------------------------------------------------------------------
# 样本命名
# ---------------------------------------------------------------------------

def make_stem(jpg_file: Path, source_root: Path, source_label: str) -> str:
    """
    为 JPG 文件生成唯一标识名。

    规则:
    - 如果图片直接在 source_root 下（无子目录），使用原始文件名
    - 如果有子目录层级，用 目录层级_文件名
    - 多数据源时，前面加上 source_label 前缀避免冲突

    示例:
        source_root=03_label_check/scene_0001
          scene_0001/0001-2027-07-29/frame_00042.jpg → "0001-2027-07-29_frame_00042"

        source_root=/abc (flat)
          /abc/frame_00001.jpg → "frame_00001"

        source_root=/abc (nested)
          /abc/sub/frame_00001.jpg → "sub_frame_00001"

        多数据源:
          src0_0001-2027-07-29_frame_00042
    """
    rel_dir = jpg_file.parent.relative_to(source_root)
    dir_str = str(rel_dir).replace("/", "_").replace("\\", "_")

    stem = jpg_file.stem
    if dir_str == ".":
        name = stem
    else:
        name = f"{dir_str}_{stem}"

    if source_label:
        name = f"{source_label}_{name}"

    return name


# ---------------------------------------------------------------------------
# 样本发现与分类
# ---------------------------------------------------------------------------

def discover_samples(source_paths: list, name_to_id: dict):
    """
    递归扫描多个数据源目录，收集所有样本，分类为正样本/空样本。

    参数:
        source_paths: 数据源路径列表
        name_to_id: {类别名: class_id} 映射

    返回: (positives, negatives)
      每个元素为 dict: {
          "stem": 唯一标识名,
          "jpg": jpg路径,
          "txt": txt路径,
          "json": json路径(可选),
      }
    """
    positives = []
    negatives = []
    total_scanned = 0
    skipped_no_label = 0
    multi_source = len(source_paths) > 1

    for src_idx, src in enumerate(source_paths):
        source = Path(src)
        if not source.exists():
            print(f"[错误] 数据源目录不存在: {source}")
            sys.exit(1)

        # 多数据源时加前缀避免文件名冲突
        source_label = f"src{src_idx}" if multi_source else ""

        print(f"\n扫描数据源 [{src_idx}]: {source}")

        for jpg_file in sorted(source.rglob("*.jpg")):
            # 跳过隐藏文件
            if jpg_file.name.startswith("."):
                continue

            total_scanned += 1
            unique_stem = make_stem(jpg_file, source, source_label)

            # 查找同名标注文件
            txt_file = jpg_file.with_suffix(".txt")
            json_file = jpg_file.with_suffix(".json")

            # 规则(2): 没有 .txt 也没有 .json → 跳过
            if not txt_file.exists() and not json_file.exists():
                print(f"  [警告] 无标注文件，跳过: {jpg_file.name}")
                skipped_no_label += 1
                continue

            sample = {
                "stem": unique_stem,
                "jpg": str(jpg_file),
                "txt": str(txt_file) if txt_file.exists() else None,
                "json": str(json_file) if json_file.exists() else None,
            }

            # 判断正/空样本
            if txt_file.exists():
                txt_content = txt_file.read_text(encoding="utf-8").strip()
                if txt_content:
                    if validate_txt(txt_content):
                        positives.append(sample)
                    else:
                        # .txt 格式无效，尝试从 JSON 重新生成
                        print(f"  [警告] .txt 格式无效，从 JSON 重新生成: {txt_file.name}")
                        if json_file.exists():
                            try:
                                regenerate_txt_from_json(sample, name_to_id)
                                positives.append(sample)
                            except Exception as e:
                                print(f"  [错误] JSON 转换失败: {e}")
                                negatives.append(sample)
                        else:
                            print(f"  [错误] 无 JSON 可恢复，归为空样本")
                            negatives.append(sample)
                else:
                    negatives.append(sample)
            elif json_file.exists():
                # 有 JSON 无 TXT —— 规则(1): 从 JSON 转换
                print(f"  [信息] 从 JSON 转换为 TXT: {json_file.name}")
                try:
                    regenerate_txt_from_json(sample, name_to_id)
                    if sample["txt"]:
                        check = Path(sample["txt"]).read_text(encoding="utf-8").strip()
                        if check:
                            positives.append(sample)
                        else:
                            negatives.append(sample)
                    else:
                        negatives.append(sample)
                except Exception as e:
                    print(f"  [错误] JSON 转换失败: {e}")
                    negatives.append(sample)
            else:
                negatives.append(sample)

    print(f"\n{'='*60}")
    print(f"扫描完成:")
    print(f"  数据源数:   {len(source_paths)}")
    print(f"  总帧数:     {total_scanned}")
    print(f"  正样本:     {len(positives)}")
    print(f"  空样本:     {len(negatives)}")
    if skipped_no_label:
        print(f"  跳过(无标注): {skipped_no_label}")
    print(f"{'='*60}")

    return positives, negatives


# ---------------------------------------------------------------------------
# TXT 格式验证
# ---------------------------------------------------------------------------

def validate_txt(content: str, n_classes: int = None) -> bool:
    """
    验证 YOLO txt 格式: 每行 class_id cx cy w h
    所有值应在 [0, 1] 范围内。空文件视为合法（空样本）。
    """
    if not content.strip():
        return True
    for line in content.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return False
        class_id = int(vals[0])
        if n_classes is not None and (class_id < 0 or class_id >= n_classes):
            return False
        for v in vals[1:]:
            if v < 0.0 or v > 1.0:
                return False
    return True


# ---------------------------------------------------------------------------
# JSON → TXT 转换
# ---------------------------------------------------------------------------

def labelme_json_to_yolo(json_path: str, txt_path: str, name_to_id: dict):
    """
    将 LabelMe JSON 标注转换为 YOLO txt 格式。
    仅处理 shape_type == "rectangle" 的标注。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")

    if not img_w or not img_h:
        raise ValueError(f"JSON 缺少 imageWidth/imageHeight")

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in name_to_id:
            print(f"    [警告] 未知类别 '{label}'，跳过")
            continue

        shape_type = shape.get("shape_type", "")
        if shape_type != "rectangle":
            print(f"    [警告] 非矩形标注 '{shape_type}'，跳过")
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            print(f"    [警告] 标注点数不足 ({len(points)})，跳过")
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cx = ((x_min + x_max) / 2) / img_w
        cy = ((y_min + y_max) / 2) / img_h
        bw = (x_max - x_min) / img_w
        bh = (y_max - y_min) / img_h

        class_id = name_to_id[label]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def regenerate_txt_from_json(sample: dict, name_to_id: dict):
    """
    从 JSON 重新生成 TXT 文件，TXT 放在 JSON 同目录下。
    """
    json_path = sample["json"]
    txt_path = str(Path(json_path).with_suffix(".txt"))
    labelme_json_to_yolo(json_path, txt_path, name_to_id)
    sample["txt"] = txt_path


# ---------------------------------------------------------------------------
# 分配逻辑
# ---------------------------------------------------------------------------

def allocate(
    positives: list,
    negatives: list,
    split: tuple,
    pos_neg_ratio: tuple,
    rounding: str,
    seed: int,
):
    """
    按比例分配正样本和空样本到 train/val/test。
    正样本用最大余数法分配，空样本按正/空比例严格控制。
    """
    rng = random.Random(seed)

    pos = list(positives)
    neg = list(negatives)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos = len(pos)
    n_neg = len(neg)

    # 最大余数法分配正样本
    pos_counts = _largest_remainder(n_pos, list(split))

    # 计算每个子集需要的空样本数
    ratio_pos, ratio_neg = pos_neg_ratio

    neg_targets = []
    for pc in pos_counts:
        if rounding == "floor":
            nt = pc * ratio_neg // ratio_pos
        else:
            nt = round(pc * ratio_neg / ratio_pos)
        neg_targets.append(nt)

    # 分配空样本
    neg_used_total = 0
    neg_assignments = []
    for nt in neg_targets:
        available = n_neg - neg_used_total
        take = min(nt, available)
        neg_assignments.append(take)
        neg_used_total += take

    if neg_used_total < sum(neg_targets):
        shortfall = sum(neg_targets) - neg_used_total
        print(f"\n[警告] 空样本不足！需要 {sum(neg_targets)} 个，可用 {n_neg} 个，短缺 {shortfall} 个")

    # 切分列表
    p0, p1 = pos_counts[0], pos_counts[0] + pos_counts[1]
    pos_train = pos[:p0]
    pos_val   = pos[p0:p1]
    pos_test  = pos[p1:]

    n0 = neg_assignments[0]
    n1 = n0 + neg_assignments[1]
    neg_train = neg[:n0]
    neg_val   = neg[n0:n1]
    neg_test  = neg[n1:n0 + neg_assignments[1] + neg_assignments[2]]

    allocation = {
        "train": {"pos": pos_train, "neg": neg_train},
        "val":   {"pos": pos_val,   "neg": neg_val},
        "test":  {"pos": pos_test,  "neg": neg_test},
    }

    # 打印分配表
    print(f"\n{'='*60}")
    print(f"数据集分配方案 (seed={seed}, rounding={rounding})")
    print(f"{'='*60}")
    print(f"{'子集':<8} {'正样本':<10} {'空样本':<10} {'合计':<10} {'实际正:空'}")
    print(f"{'-'*60}")

    for label in ["train", "val", "test"]:
        pc = len(allocation[label]["pos"])
        nc = len(allocation[label]["neg"])
        total = pc + nc
        if nc > 0:
            actual = f"{pc}:{nc}"
        else:
            actual = f"{pc}:0 (全正样本)"
        print(f"{label:<8} {pc:<10} {nc:<10} {total:<10} {actual}")

    print(f"{'-'*60}")
    print(f"{'总计':<8} {n_pos:<10} {neg_used_total:<10} {n_pos + neg_used_total:<10}")
    print(f"{'='*60}")

    unused = n_neg - neg_used_total
    if unused > 0:
        print(f"未使用的空样本: {unused} 个")

    return allocation


def _largest_remainder(n: int, ratios: list) -> list:
    """最大余数法分配整数。"""
    total = sum(ratios)
    quotas = [n * r / total for r in ratios]
    floors = [int(math.floor(q)) for q in quotas]
    remainders = [(q - f, i) for i, (q, f) in enumerate(zip(quotas, floors))]
    remainders.sort(key=lambda x: x[0], reverse=True)
    remaining = n - sum(floors)
    result = list(floors)
    for _, idx in remainders[:remaining]:
        result[idx] += 1
    return result


# ---------------------------------------------------------------------------
# 写入数据集
# ---------------------------------------------------------------------------

def write_dataset(
    allocation: dict,
    out_dir: str,
    class_names: list,
    args: argparse.Namespace,
):
    """将分配好的数据集写入磁盘。"""
    out = Path(out_dir)
    if out.exists():
        if args.overwrite:
            print(f"\n[信息] 删除已存在的输出目录: {out}")
            shutil.rmtree(out)
        else:
            print(f"\n[错误] 输出目录已存在: {out}")
            print("  使用 --overwrite 覆盖，或指定其他 --name")
            sys.exit(1)

    splits = ["train", "val", "test"]
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    copied = 0
    for split_name in splits:
        samples = allocation[split_name]["pos"] + allocation[split_name]["neg"]
        for sample in samples:
            stem = sample["stem"]

            jpg_src = Path(sample["jpg"])
            jpg_dst = out / "images" / split_name / f"{stem}.jpg"
            shutil.copy2(jpg_src, jpg_dst)

            txt_dst = out / "labels" / split_name / f"{stem}.txt"
            if sample["txt"] and Path(sample["txt"]).exists():
                shutil.copy2(sample["txt"], txt_dst)
            else:
                txt_dst.write_text("", encoding="utf-8")

            copied += 1

    print(f"\n写入完成: 共 {copied} 个文件")
    write_data_yaml(out, class_names)
    write_stats(out, allocation, class_names, args)


def write_data_yaml(out_dir: Path, class_names: list):
    """生成 data.yaml。"""
    lines = [
        "path: .",
        "",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  配置: {yaml_path}")


def write_stats(
    out_dir: Path,
    allocation: dict,
    class_names: list,
    args: argparse.Namespace,
):
    """写入 dataset_stats.txt 统计文件。"""
    stats_path = out_dir / "dataset_stats.txt"
    sources_str = ", ".join(args.source)

    lines = [
        f"数据集名称: {args.name}",
        f"随机种子:    {args.seed}",
        f"取整方式:    {args.rounding}",
        "",
        "参数:",
        f"  数据源:     {sources_str}",
        f"  类别文件:   {args.classes}",
        f"  拆分比例:   {args.split[0]}:{args.split[1]}:{args.split[2]} (train:val:test)",
        f"  正/空比例:  {args.pos_neg_ratio[0]}:{args.pos_neg_ratio[1]}",
        "",
        f"类别 ({len(class_names)} 个):",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    lines.append("")
    lines.append(f"{'子集':<8} {'正样本':<10} {'空样本':<10} {'合计':<10}")
    lines.append(f"{'-'*40}")

    total_pos = 0
    total_neg = 0
    for label in ["train", "val", "test"]:
        pc = len(allocation[label]["pos"])
        nc = len(allocation[label]["neg"])
        total_pos += pc
        total_neg += nc
        lines.append(f"{label:<8} {pc:<10} {nc:<10} {pc + nc:<10}")

    lines.append(f"{'-'*40}")
    lines.append(f"{'总计':<8} {total_pos:<10} {total_neg:<10} {total_pos + total_neg:<10}")

    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  统计: {stats_path}")


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

def verify(out_dir: str):
    """验证输出数据集的完整性：图片标注一一对应，标注格式正确。"""
    out = Path(out_dir)
    errors = []

    print(f"\n{'='*60}")
    print("验证数据集")
    print(f"{'='*60}")

    for split_name in ["train", "val", "test"]:
        img_dir = out / "images" / split_name
        lbl_dir = out / "labels" / split_name

        if not img_dir.exists():
            continue

        jpgs = set(p.stem for p in img_dir.glob("*.jpg") if not p.name.startswith("."))
        txts = set(p.stem for p in lbl_dir.glob("*.txt") if not p.name.startswith("."))

        only_jpg = jpgs - txts
        only_txt = txts - jpgs

        if only_jpg:
            errors.append(f"{split_name}: 有图片无标注 -> {only_jpg}")
        if only_txt:
            errors.append(f"{split_name}: 有标注无图片 -> {only_txt}")

        print(f"  {split_name}: {len(jpgs)} 张图片, {len(txts)} 个标注")

        for txt_file in sorted(lbl_dir.glob("*.txt"))[:3]:
            content = txt_file.read_text(encoding="utf-8").strip()
            if content and not validate_txt(content):
                errors.append(f"{split_name}: 格式异常 -> {txt_file.name}")

    if not (out / "data.yaml").exists():
        errors.append("缺少 data.yaml")

    if errors:
        print(f"\n[错误] 发现 {len(errors)} 个问题:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\n验证通过 ✓")
        return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("YOLO 训练数据集创建工具")
    print("=" * 60)
    print(f"数据源:     {', '.join(args.source)}")
    print(f"数据集名:   {args.name}")
    print(f"输出目录:   {args.dataset_dir}")
    print(f"类别文件:   {args.classes}")
    print(f"拆分比例:   {args.split[0]}:{args.split[1]}:{args.split[2]} (train:val:test)")
    print(f"正/空比例:  {args.pos_neg_ratio[0]}:{args.pos_neg_ratio[1]}")
    print(f"取整方式:   {args.rounding}")
    print(f"随机种子:   {args.seed}")
    print()

    # 1. 加载类别
    class_names = load_classes(args.classes)
    if not class_names:
        print("[错误] 类别文件为空或格式错误")
        sys.exit(1)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    print(f"加载类别 ({len(class_names)} 个): {class_names}")

    # 2. 扫描数据源
    positives, negatives = discover_samples(args.source, name_to_id)

    if not positives and not negatives:
        print("[错误] 数据源中未找到任何样本")
        sys.exit(1)

    if not positives:
        print("[错误] 数据源中未找到正样本，无法创建数据集")
        sys.exit(1)

    # 3. 分配
    allocation = allocate(
        positives=positives,
        negatives=negatives,
        split=tuple(args.split),
        pos_neg_ratio=tuple(args.pos_neg_ratio),
        rounding=args.rounding,
        seed=args.seed,
    )

    # 4. 输出
    out_dir = os.path.join(args.dataset_dir, args.name)

    if args.dry_run:
        print(f"\n[干运行] 不会写入任何文件")
        print(f"目标目录: {out_dir}")
        return

    print(f"\n写入数据集到: {out_dir}")
    write_dataset(allocation, out_dir, class_names, args)

    # 5. 验证
    ok = verify(out_dir)
    if not ok:
        sys.exit(1)

    print(f"\n完成！数据集已创建: {out_dir}")


if __name__ == "__main__":
    main()
