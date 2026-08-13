#!/usr/bin/env python3
"""
将 YOLO 训练集的标注框绘制到图片上，输出可视化结果。

用法:
    python scripts/draw_bbox.py -s Root/04_dataset/0002_20270812 -o output/vis

    # 只处理 train 子集
    python scripts/draw_bbox.py -s Root/04_dataset/0002_20270812 -o output/vis --splits train

    # 处理所有子集
    python scripts/draw_bbox.py -s Root/04_dataset/0002_20270812 -o output/vis --splits train val test
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    sys.exit(
        "错误: 缺少 Pillow 库\n\n"
        "安装方法:\n"
        "  # 方式一: 使用虚拟环境 (推荐)\n"
        "  python3 -m venv venv\n"
        "  source venv/bin/activate\n"
        "  pip install Pillow\n"
        "  python scripts/draw_bbox.py ...\n\n"
        "  # 方式二: 直接安装 (可能需 --user 或 --break-system-packages)\n"
        "  pip install Pillow\n"
    )


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在训练集图片上绘制 YOLO 标注框"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="数据集路径（包含 images/ 和 labels/ 子目录）",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="要处理的子集 (默认: train val)",
    )
    parser.add_argument(
        "--classes", "-c",
        default=None,
        help="类别文件路径（classes.txt），用于显示类别名。不指定则显示 class_id",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="边框线宽 (默认: 3)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=20,
        help="标签字号 (默认: 20)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 类别
# ---------------------------------------------------------------------------

def load_classes(path: str) -> list:
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


# ---------------------------------------------------------------------------
# 颜色
# ---------------------------------------------------------------------------

# 预设 20 种高对比度颜色
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (192, 192, 192), (64, 64, 64),
]


def get_color(class_id: int) -> tuple:
    return COLORS[class_id % len(COLORS)]


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------

def draw_boxes(
    img_path: Path,
    label_path: Path,
    class_names: list,
    line_width: int,
    font_size: int,
) -> Image.Image:
    """在图片上绘制 YOLO 标注框，返回带标注的 Image 对象。"""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # 尝试加载字体，失败则用默认
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    if not label_path.exists():
        return img

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return img  # 空样本，无标注框

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(x) for x in parts[1:]]

        # YOLO 归一化坐标 → 像素坐标
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = get_color(class_id)

        # 画矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 类别标签
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = str(class_id)

        # 标签背景
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        label_y = y1 - th - 4
        if label_y < 0:
            label_y = y1 + 2

        draw.rectangle([x1, label_y, x1 + tw + 4, label_y + th + 4], fill=color)
        draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255), font=font)

    return img


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    # 类别
    class_names = None
    if args.classes:
        class_names = load_classes(args.classes)
        print(f"加载类别 ({len(class_names)} 个): {class_names}")
    else:
        print("未指定类别文件，标注将使用 class_id")

    total_done = 0
    total_boxes = 0

    for split_name in args.splits:
        img_dir = source / "images" / split_name
        lbl_dir = source / "labels" / split_name

        if not img_dir.exists():
            print(f"\n[跳过] 目录不存在: {img_dir}")
            continue

        # 收集所有 jpg
        jpg_files = sorted(
            p for p in img_dir.glob("*.jpg") if not p.name.startswith(".")
        )
        if not jpg_files:
            print(f"\n[跳过] {split_name}: 无图片")
            continue

        # 输出子目录
        out_dir = output / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        n = len(jpg_files)
        print(f"\n{'='*60}")
        print(f"[{split_name}] {n} 张图片")
        print(f"{'='*60}")

        # 进度步长（每 10% 或最少 1 张报告一次）
        report_step = max(1, n // 10)

        for idx, jpg_file in enumerate(jpg_files):
            stem = jpg_file.stem
            lbl_file = lbl_dir / f"{stem}.txt"

            # 统计标注框数
            n_boxes = 0
            if lbl_file.exists():
                content = lbl_file.read_text(encoding="utf-8").strip()
                if content:
                    n_boxes = len(content.splitlines())

            # 绘制
            annotated = draw_boxes(
                jpg_file, lbl_file, class_names,
                args.line_width, args.font_size,
            )

            # 保存
            out_path = out_dir / f"{stem}.jpg"
            annotated.save(out_path, quality=95)

            total_done += 1
            total_boxes += n_boxes

            # 进度
            if (idx + 1) % report_step == 0 or idx == n - 1:
                pct = (idx + 1) * 100 // n
                print(f"  [{split_name}] {idx + 1}/{n} ({pct}%)  累计: {total_done} 张, {total_boxes} 个框")

    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"  处理图片: {total_done} 张")
    print(f"  绘制标注框: {total_boxes} 个")
    print(f"  输出目录: {output.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
