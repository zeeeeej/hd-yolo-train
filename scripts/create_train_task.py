#!/usr/bin/env python3
"""
YOLO 训练任务生成脚本

根据数据集创建训练任务目录，生成训练脚本和运维脚本（start/stop/hdlog），
最终打包为 .tar.gz 上传到训练服务器使用。

用法:
    python scripts/create_train_task.py \
      --dataset Root/04_dataset/0002_20270812 \
      --name 0002-20260812-yolov11n

    python scripts/create_train_task.py \
      --dataset Root/04_dataset/0002_20270812 \
      --name 0002-20260812-yolov11n \
      --epochs 300 --batch 32 --output-tar
"""

import argparse
import os
import shutil
import stat
import sys
import tarfile
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="创建 YOLO 训练任务包"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="数据集路径（04_dataset 下的目录，或 .tar.gz 文件）",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="训练任务名称 (如: 0002-20260812-yolov11n)",
    )
    parser.add_argument(
        "--train-dir",
        default=str(script_dir / "Root" / "05_train"),
        help="训练任务输出目录 (默认: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="yolo11n",
        help="YOLO 模型名 (默认: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="默认训练轮数 (默认: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="数据加载线程数 (默认: %(default)s)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=24,
        help="批次大小 (默认: %(default)s)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸 (默认: %(default)s)",
    )

    # 服务器路径配置（脚本中作为可配置变量）
    parser.add_argument(
        "--conda-path",
        default="/root/miniconda3/etc/profile.d/conda.sh",
        help="服务器 conda 路径 (默认: %(default)s)",
    )
    parser.add_argument(
        "--conda-env",
        default="rv1106-ultralytics",
        help="conda 环境名 (默认: %(default)s)",
    )
    parser.add_argument(
        "--ultralytics-path",
        default="/root/ultralytics-8.3.39-rv1106",
        help="服务器 ultralytics 路径 (默认: %(default)s)",
    )
    parser.add_argument(
        "--output-tar",
        action="store_true",
        default=True,
        help="生成 .tar.gz 打包 (默认: 是)",
    )
    parser.add_argument(
        "--no-tar",
        action="store_true",
        help="不生成 .tar.gz",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的任务目录",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据集处理
# ---------------------------------------------------------------------------

def prepare_dataset(dataset_path: str, task_dir: Path) -> str:
    """
    将数据集复制/解压到任务目录下的 dataset/ 目录。

    返回原数据集名（如 0002_20270812，用于日志命名等）
    """
    src = Path(dataset_path)
    if not src.exists():
        print(f"[错误] 数据集不存在: {src}")
        sys.exit(1)

    if src.is_dir():
        # 目录 → 复制为 dataset/
        dataset_name = src.name
        dst_dir = task_dir / "dataset"
        print(f"复制数据集: {src} → dataset/")
        shutil.copytree(src, dst_dir)
        _clean_macos_junk(dst_dir)
        return dataset_name

    elif src.suffix == ".gz" or src.suffix == ".tar" or ".tar." in src.name:
        # tar.gz → 解压后重命名为 dataset/
        dataset_name = src.name.replace(".tar.gz", "").replace(".tar", "")
        dst_dir = task_dir / "dataset"
        print(f"解压数据集: {src.name} → dataset/")
        with tarfile.open(src, "r:gz") as tar:
            tar.extractall(task_dir)
        extracted = task_dir / dataset_name
        if extracted.exists() and extracted != dst_dir:
            extracted.rename(dst_dir)
        _clean_macos_junk(dst_dir)
        return dataset_name

    else:
        print(f"[错误] 不支持的数据集格式: {src}")
        sys.exit(1)


def _clean_macos_junk(directory: Path):
    """清理 macOS 产生的 ._ 前缀文件和 .DS_Store"""
    for junk in directory.rglob("._*"):
        junk.unlink()
    for junk in directory.rglob(".DS_Store"):
        junk.unlink()


# ---------------------------------------------------------------------------
# 脚本模板
# ---------------------------------------------------------------------------

def generate_train_sh(args) -> str:
    """生成 train_rv1106_bz.sh 内容"""
    return f'''#!/bin/bash
# ====== 可配置变量 ======
CONDA_PATH="{args.conda_path}"
CONDA_ENV="{args.conda_env}"
ULTRALYTICS_PATH="{args.ultralytics_path}"
MODEL="{args.model}.pt"
WORKERS={args.workers}
BATCH={args.batch}
IMGSZ={args.imgsz}
# ========================

ver=$1
epochs=$2

# 转为绝对路径，确保 yolo 能正确解析 data.yaml 中的 path
VER_ABS=$(realpath "${{ver}}" 2>/dev/null || readlink -f "${{ver}}" 2>/dev/null || echo "$(pwd)/${{ver}}")
DATA_DIR=$(dirname "${{VER_ABS}}")

echo "=== train_rv1106_bz.sh ==="
echo "  data.yaml: ${{VER_ABS}}"
echo "  data dir:  ${{DATA_DIR}}"
echo "  epochs:    ${{epochs}}"

source ${{CONDA_PATH}}
conda activate ${{CONDA_ENV}}

# cd 到数据集目录，path: . 正确指向 images/ 和 labels/
cd "${{DATA_DIR}}" && \\
yolo detect train data="${{VER_ABS}}" model=${{ULTRALYTICS_PATH}}/${{MODEL}} epochs=${{epochs}} workers=${{WORKERS}} batch=${{BATCH}} imgsz=${{IMGSZ}}
'''


def generate_execute_sh(args, dataset_name: str) -> str:
    """生成 train_rv1106_bz_execute.sh 内容"""
    return f'''#!/bin/bash
# ====== 可配置变量 ======
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
LOGS_DIR="${{TASK_DIR}}/logs"
DATASET_NAME="{dataset_name}"
# ========================

modelVersionId=$1
epochs=$2

TIME=$(/bin/date +%Y%m%d%H%M%S)
LOG_FILE="${{LOGS_DIR}}/train_${{DATASET_NAME}}_${{TIME}}.log"

echo "=== hadlinks train start ==="
echo "  data.yaml: ${{modelVersionId}}"
echo "  epochs:    ${{epochs}}"
echo "  log:       ${{LOG_FILE}}"

mkdir -p ${{LOGS_DIR}}

nohup ${{SCRIPT_DIR}}/train_rv1106_bz.sh ${{modelVersionId}} ${{epochs}} \\
  >> ${{LOG_FILE}} 2>&1 &
'''


def generate_start_sh(dataset_name: str, args) -> str:
    """生成 start.sh 内容"""
    return f'''#!/bin/bash
# ====== 可配置变量 ======
DATASET_DIR="./dataset"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# ========================

# 支持命令行指定 epochs: ./start.sh 300
EPOCHS=${{1:-{args.epochs}}}

DATA_YAML="${{DATASET_DIR}}/data.yaml"

if [ ! -f "${{DATA_YAML}}" ]; then
    echo "[错误] data.yaml 不存在: ${{DATA_YAML}}"
    exit 1
fi

echo "========================================="
echo "  启动 YOLO 训练"
echo "========================================="
echo "  数据集:   ${{DATASET_DIR}}"
echo "  data.yaml: ${{DATA_YAML}}"
echo "  epochs:   ${{EPOCHS}}"
echo "========================================="
echo ""

cd "${{SCRIPT_DIR}}"
./script/train_rv1106_bz_execute.sh "${{DATA_YAML}}" ${{EPOCHS}}

echo ""
echo "训练已启动，使用 ./hdlog.sh 查看日志"
'''


def generate_hdlog_sh(args) -> str:
    """生成 hdlog.sh 内容"""
    return '''#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"

LATEST=$(ls -t ${LOGS_DIR}/train_*.log 2>/dev/null | head -1)

if [ -n "$LATEST" ]; then
    echo "实时查看: ${LATEST}"
    tail -n 200 -f "${LATEST}"
else
    echo "未找到训练日志 (${LOGS_DIR}/train_*.log)"
    echo "请确认训练已启动: ./start.sh"
fi
'''


def generate_stop_sh() -> str:
    """生成 stop.sh 内容"""
    return '''#!/bin/bash
echo "=== 查找训练进程 ==="
ps aux | grep "yolo detect train" | grep -v grep
echo ""
ps aux | grep "ultralytics" | grep -v grep

PIDS=$(ps aux | grep "yolo detect train" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo ""
    echo "未找到运行中的训练进程"
    exit 0
fi

echo ""
echo "找到进程 PID: $PIDS"
echo "正在发送 SIGTERM (15) 优雅终止..."
for pid in $PIDS; do
    kill -15 $pid 2>/dev/null && echo "  已终止 PID: $pid" || echo "  终止失败 PID: $pid"
done
echo ""
echo "训练已停止"
'''


def generate_pack_result_sh(dataset_name: str) -> str:
    """生成 pack_result.sh 内容 —— 打包训练结果

    注意（CLAUDE.md 铁律）: 不删除任何文件，输出用时间戳命名
    """
    return f'''#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="${{SCRIPT_DIR}}/dataset"
LOGS_DIR="${{SCRIPT_DIR}}/logs"

# 时间戳命名，不覆盖/删除已有文件
STAMP=$(/bin/date +%Y%m%d%H%M%S)
OUTPUT="${{SCRIPT_DIR}}/result_${{STAMP}}.tar.gz"

echo "=== 打包训练结果 ==="

# 找到最新的 runs/detect/train* 目录
RESULT_DIR=$(ls -td "${{DATASET_DIR}}"/runs/detect/train* 2>/dev/null | head -1)

if [ -z "${{RESULT_DIR}}" ]; then
    echo "[错误] 未找到训练结果: ${{DATASET_DIR}}/runs/detect/train*"
    echo "请先完成训练 (./start.sh)"
    exit 1
fi

echo "  结果目录: ${{RESULT_DIR}}"
echo "  日志目录: ${{LOGS_DIR}}"
echo "  输出文件: ${{OUTPUT}}"
echo ""

# 打包 train 目录 + 训练日志（tar 创建新文件，不删除旧包）
cd "${{DATASET_DIR}}/runs/detect"
tar czf "${{OUTPUT}}" \\
    "$(basename "${{RESULT_DIR}}")" \\
    -C "${{SCRIPT_DIR}}" logs

SIZE=$(du -h "${{OUTPUT}}" | cut -f1)
echo "完成！${{OUTPUT}} (${{SIZE}})"
echo ""
echo "已有的结果包:"
ls -lh "${{SCRIPT_DIR}}"/result_*.tar.gz 2>/dev/null
'''


def generate_readme(dataset_name: str, task_name: str, args) -> str:
    """生成 README.txt 内容"""
    return f"""============================================
  YOLO 训练任务: {task_name}
============================================

数据集: {dataset_name}
模型:   {args.model}.pt
默认 epochs: {args.epochs}
workers: {args.workers}
batch:   {args.batch}
imgsz:   {args.imgsz}

---- 使用说明 ----

1. 启动训练:
   ./start.sh           # 使用默认 epochs
   ./start.sh 300       # 指定 epochs

2. 查看日志:
   ./hdlog.sh

3. 停止训练:
   ./stop.sh

4. 打包训练结果:
   ./pack_result.sh       # 生成 result_时间戳.tar.gz（不覆盖旧包）

---- 目录结构 ----
{task_name}/
├── dataset/                  # 数据集（原 {dataset_name}，训练结果在 runs/ 下）
├── yolo26n.pt                # 公共模型（防止服务器 AMP 检查联网下载）
├── logs/                     # 训练日志
├── start.sh                  # 启动训练
├── hdlog.sh                  # 查看日志
├── stop.sh                   # 停止训练
├── pack_result.sh            # 打包训练结果 → result_时间戳.tar.gz
└── script/
    ├── train_rv1106_bz.sh         # 训练核心
    └── train_rv1106_bz_execute.sh # nohup 后台启动
"""


# ---------------------------------------------------------------------------
# 写入脚本
# ---------------------------------------------------------------------------

def write_script(task_dir: Path, filename: str, content: str):
    """写入脚本文件并设置可执行权限"""
    path = task_dir / filename
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  写入: {filename}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    task_dir = Path(args.train_dir) / args.name

    print("=" * 60)
    print("YOLO 训练任务生成工具")
    print("=" * 60)
    print(f"数据集:     {args.dataset}")
    print(f"任务名称:   {args.name}")
    print(f"输出目录:   {args.train_dir}")
    print(f"模型:       {args.model}.pt")
    print(f"epochs:     {args.epochs}")
    print(f"workers:    {args.workers}")
    print(f"batch:      {args.batch}")
    print(f"imgsz:      {args.imgsz}")
    print()

    # 检查输出目录
    if task_dir.exists():
        if args.overwrite:
            print(f"[信息] 删除已存在的任务目录: {task_dir}")
            shutil.rmtree(task_dir)
        else:
            print(f"[错误] 任务目录已存在: {task_dir}")
            print("  使用 --overwrite 覆盖，或指定其他 --name")
            sys.exit(1)

    # 创建目录结构
    script_dir = task_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建任务目录: {task_dir}")

    # 1. 处理数据集
    print(f"\n--- 数据集 ---")
    dataset_name = prepare_dataset(args.dataset, task_dir)

    # 2. 生成脚本
    print(f"\n--- 生成脚本 ---")
    # script/ 下
    write_script(script_dir, "train_rv1106_bz.sh", generate_train_sh(args))
    write_script(script_dir, "train_rv1106_bz_execute.sh", generate_execute_sh(args, dataset_name))
    # 任务根目录下
    write_script(task_dir, "start.sh", generate_start_sh(dataset_name, args))
    write_script(task_dir, "hdlog.sh", generate_hdlog_sh(args))
    write_script(task_dir, "stop.sh", generate_stop_sh())
    write_script(task_dir, "pack_result.sh", generate_pack_result_sh(dataset_name))

    # 3. 公共模型 yolo26n.pt
    #    任务根目录放一份（用户要求，与 script/ 平级）
    #    dataset/ 再放一份 —— 训练进程 CWD 是 dataset/，ultralytics 只在 CWD 查找，
    #    这一份才能真正阻止 AMP 检查时从 GitHub 下载
    print(f"\n--- 公共模型 ---")
    yolo26n_src = Path(__file__).resolve().parent.parent / "model" / "yolo26n.pt"
    if yolo26n_src.exists():
        shutil.copy2(yolo26n_src, task_dir / "yolo26n.pt")
        print(f"  复制: → {task_dir.name}/yolo26n.pt")
        shutil.copy2(yolo26n_src, task_dir / "dataset" / "yolo26n.pt")
        print(f"  复制: → {task_dir.name}/dataset/yolo26n.pt (训练 CWD，阻止 AMP 下载)")
    else:
        print(f"  [警告] 未找到 {yolo26n_src}，跳过（服务器训练时可能需要联网下载）")

    # 4. README
    readme = generate_readme(dataset_name, args.name, args)
    (task_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(f"  写入: README.txt")

    # 5. 打包 tar.gz
    if args.no_tar:
        print(f"\n[跳过] 不生成 .tar.gz")
    else:
        print(f"\n--- 打包 ---")
        tar_path = Path(args.train_dir) / f"{args.name}.tar.gz"
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(task_dir), arcname=args.name)
        print(f"  已生成: {tar_path}")

        # 显示大小
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"  大小: {size_mb:.1f} MB")

    print(f"\n{'='*60}")
    print(f"完成！训练任务: {task_dir}")
    if not args.no_tar:
        print(f"打包文件:     {tar_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
