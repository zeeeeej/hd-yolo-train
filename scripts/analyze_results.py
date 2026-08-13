#!/usr/bin/env python3
"""
训练结果分析脚本

分析单个或多个训练任务的结果（Root/05_train/{task}/ 下的 result_*.tar.gz），
输出指标曲线图、训练产物图、汇总报告到 Root/06_data_analysis/{自动命名}/。

用法:
    # 分析单个任务
    python scripts/analyze_results.py --tasks 0005-20260813-yolov11n

    # 对比多个任务（明确提出才对比）
    python scripts/analyze_results.py --tasks 0005-20260813-yolov11n 0004-20260813-yolov11s

输出结构:
    06_data_analysis/0005_vs_0004_20260813/
    ├── summary.md             # 汇总报告（指标表 + 数据集信息）
    ├── charts/                # 指标曲线图（loss/mAP 随 epoch 变化）
    ├── artifacts/{task}/      # 训练产物图（PR曲线/混淆矩阵/批次样本等）
    └── data/{task}_results.csv

依赖: pip install matplotlib
"""

import argparse
import csv
import io
import re
import sys
import tarfile
from datetime import date
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，直接输出 PNG
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    sys.exit("错误: 缺少 matplotlib 库，请先安装:\n\n  pip install matplotlib\n")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_DIR = PROJECT_ROOT / "Root" / "05_train"
ANALYSIS_DIR = PROJECT_ROOT / "Root" / "06_data_analysis"

# 参考配色（已验证调色板，固定槽位顺序: 蓝/橙/青）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# results.csv 指标列 → 图表显示名
METRICS = [
    ("train/box_loss", "train/box_loss"),
    ("train/cls_loss", "train/cls_loss"),
    ("train/dfl_loss", "train/dfl_loss"),
    ("val/box_loss", "val/box_loss"),
    ("val/cls_loss", "val/cls_loss"),
    ("val/dfl_loss", "val/dfl_loss"),
    ("metrics/precision(B)", "precision"),
    ("metrics/recall(B)", "recall"),
    ("metrics/mAP50(B)", "mAP50"),
    ("metrics/mAP50-95(B)", "mAP50-95"),
]

# 训练产物图（从 result tar.gz 中提取）
ARTIFACT_FILES = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "val_batch0_pred.jpg",
    "val_batch0_labels.jpg",
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def find_result_tar(task_name: str) -> Path:
    """找到任务目录下最新的 result_*.tar.gz"""
    task_dir = TRAIN_DIR / task_name
    if not task_dir.exists():
        sys.exit(f"[错误] 任务目录不存在: {task_dir}")
    tars = sorted(
        p for p in task_dir.glob("result_*.tar.gz") if not p.name.startswith("._")
    )
    if not tars:
        sys.exit(
            f"[错误] 未找到训练结果包: {task_dir}/result_*.tar.gz\n"
            f"请先在服务器上运行 ./pack_result.sh 并把结果下载到该目录"
        )
    return tars[-1]  # 最新


def load_results_csv(tar_path: Path) -> dict:
    """从 result tar.gz 读取 train/results.csv，返回 {列名: [值...]}"""
    with tarfile.open(tar_path, "r:gz") as tar:
        member = None
        for m in tar.getmembers():
            if m.name.endswith("train/results.csv") or m.name == "train/results.csv":
                member = m
                break
        if member is None:
            sys.exit(f"[错误] {tar_path} 中没有 train/results.csv")
        raw = tar.extractfile(member).read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(raw))
    columns = {}
    for row in reader:
        for key, val in row.items():
            key = key.strip()
            try:
                columns.setdefault(key, []).append(float(val))
            except (TypeError, ValueError):
                pass
    return columns


def load_dataset_info(task_name: str) -> dict:
    """从任务目录的 dataset/dataset_stats.txt 读取数据集信息"""
    info = {"scenes": [], "counts": {}}
    stats_file = TRAIN_DIR / task_name / "dataset" / "dataset_stats.txt"
    if not stats_file.exists():
        return info
    text = stats_file.read_text(encoding="utf-8")
    # 数据源行: 数据源:     /path/scene_0001, /path/scene_0002
    for m in re.finditer(r"数据源:\s*(.+)", text):
        for src in m.group(1).split(","):
            src = src.strip()
            if src:
                # 提取场景名（scene_XXXX 或目录名）
                name = Path(src).name
                if name and name not in info["scenes"]:
                    info["scenes"].append(name)
    # 各子集数量
    for m in re.finditer(r"(train|val|test)\s+(\d+)\s+(\d+)\s+(\d+)", text):
        split, pos, neg, total = m.group(1), m.group(2), m.group(3), m.group(4)
        info["counts"][split] = f"正 {pos} / 空 {neg} / 共 {total}"
    return info


def load_args_yaml(tar_path: Path) -> dict:
    """从 result tar.gz 读取 train/args.yaml，提取关键训练参数"""
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.name.endswith("train/args.yaml") or m.name == "train/args.yaml":
                raw = tar.extractfile(m).read().decode("utf-8", errors="replace")
                params = {}
                for line in raw.splitlines():
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        k, v = line.split(":", 1)
                        params[k.strip()] = v.strip()
                return params
    return {}


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def style_ax(ax):
    """统一图表样式: 浅色表面、细网格、muted 轴线"""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_metric(metric_key: str, metric_label: str, task_names: list,
                data: dict, out_path: Path):
    """一个指标一张图，多任务叠加对比（一条 2px 线）"""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for i, task in enumerate(task_names):
        cols = data[task]
        if metric_key not in cols:
            print(f"  [警告] {task} 缺少指标列 {metric_key}，跳过")
            continue
        values = cols[metric_key]
        epochs = list(range(1, len(values) + 1))
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(epochs, values, linewidth=2, color=color, label=task,
                solid_capstyle="round")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label, color=INK, fontsize=11)
    if len(task_names) > 1:
        ax.legend(frameon=False, fontsize=8, loc="best")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析训练任务结果")
    parser.add_argument(
        "--tasks", "-t", nargs="+", required=True,
        help="训练任务名（可多个），如 0005-20260813-yolov11n",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="手动指定输出目录名（默认自动命名: 任务A_vs_任务B_日期）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = args.tasks

    # 输出目录自动命名
    if args.out_dir:
        out_name = args.out_dir
    elif len(tasks) == 1:
        out_name = f"{tasks[0]}_analysis_{date.today():%Y%m%d}"
    else:
        out_name = "_vs_".join(tasks) + f"_{date.today():%Y%m%d}"
    out_dir = ANALYSIS_DIR / out_name

    print("=" * 60)
    print("训练结果分析")
    print("=" * 60)
    print(f"任务: {', '.join(tasks)}")
    print(f"输出: {out_dir}")
    print()

    # 1. 加载数据
    print("--- 加载数据 ---")
    tar_paths = {}
    data = {}
    for task in tasks:
        tar = find_result_tar(task)
        tar_paths[task] = tar
        data[task] = load_results_csv(tar)
        n_epochs = len(next(iter(data[task].values()), []))
        print(f"  {task}: {tar.name} ({n_epochs} epochs)")

    # 2. 生成指标曲线图
    print("\n--- 指标曲线图 ---")
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for metric_key, metric_label in METRICS:
        safe_name = metric_label.replace("/", "_")
        plot_metric(metric_key, metric_label, tasks, data,
                    charts_dir / f"{safe_name}.png")
    print(f"  已生成 {len(METRICS)} 张: {charts_dir}")

    # 3. 提取训练产物图
    print("\n--- 训练产物图 ---")
    for task in tasks:
        art_dir = out_dir / "artifacts" / task
        art_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_paths[task], "r:gz") as tar:
            members = {m.name: m for m in tar.getmembers()}
            for fname in ARTIFACT_FILES:
                # 匹配 train/{fname}
                target = f"train/{fname}"
                m = members.get(target)
                if m is None:
                    # 兼容其他层级
                    for name, mm in members.items():
                        if name.endswith(f"/{fname}") or name == fname:
                            m = mm
                            break
                if m:
                    dest = art_dir / fname
                    dest.write_bytes(tar.extractfile(m).read())
        print(f"  {task}: {art_dir}")

    # 4. 保存 CSV 数据
    print("\n--- 数据副本 ---")
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        with tarfile.open(tar_paths[task], "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.endswith("train/results.csv"):
                    (data_dir / f"{task}_results.csv").write_bytes(
                        tar.extractfile(m).read())
                    break
    print(f"  {data_dir}")

    # 5. 汇总报告
    print("\n--- 汇总报告 ---")
    write_summary(out_dir, tasks, data, tar_paths)
    print(f"  已生成: {out_dir / 'summary.md'}")

    print("\n" + "=" * 60)
    print(f"完成！分析结果: {out_dir}")
    print("=" * 60)


def analyze_trend(cols: dict) -> dict:
    """从 results.csv 列数据提取趋势信息"""
    t = {"epochs": 0, "map50": None, "map5095": None, "box_loss": None,
         "last20_mean": None, "last20_max": None, "stable": False}
    map50 = cols.get("metrics/mAP50(B)")
    map5095 = cols.get("metrics/mAP50-95(B)")
    box_loss = cols.get("train/box_loss")
    if not map50:
        return t
    t["epochs"] = len(map50)
    t["map50"] = map50[-1]
    t["map5095"] = map5095[-1] if map5095 else None
    t["box_loss"] = box_loss[-1] if box_loss else None
    tail = map50[-20:] if len(map50) >= 20 else map50
    t["last20_mean"] = sum(tail) / len(tail)
    t["last20_max"] = max(tail)
    # 稳定判定: 最后20轮均值达到最高值的 95% 以上（波动小 = 收敛）
    if t["last20_max"] > 0:
        t["stable"] = t["last20_mean"] / t["last20_max"] >= 0.95
    return t


def make_recommendations(task: str, t: dict) -> str:
    """根据趋势数据生成一句话建议"""
    tips = []
    if t["stable"]:
        tips.append("曲线稳定收敛，模型可直接使用")
    else:
        tips.append("曲线仍在波动/爬升，训练未充分收敛")
    if t["box_loss"] is not None and t["box_loss"] > 0.7:
        tips.append("box_loss 仍偏高，建议增加 epochs（200-300）")
    return "；".join(tips)


def write_summary(out_dir: Path, tasks: list, data: dict, tar_paths: dict):
    """生成 summary.md"""
    lines = [
        f"# 训练结果分析报告",
        f"",
        f"- 生成日期: {date.today():%Y-%m-%d}",
        f"- 分析任务: {', '.join(tasks)}",
        f"",
        f"## 任务概览",
        f"",
        f"| 任务 | 模型 | epochs | 数据集场景 | 训练集 | 验证集 | 测试集 |",
        f"|------|------|--------|-----------|--------|--------|--------|",
    ]
    for task in tasks:
        args_yaml = load_args_yaml(tar_paths[task])
        info = load_dataset_info(task)
        model = Path(args_yaml.get("model", "-")).name  # 只显示文件名，路径是服务器路径
        epochs = args_yaml.get("epochs", "-")
        scenes = ", ".join(info["scenes"]) if info["scenes"] else "-"
        counts = info.get("counts", {})
        lines.append(
            f"| {task} | {model} | {epochs} | {scenes} | "
            f"{counts.get('train', '-')} | {counts.get('val', '-')} | "
            f"{counts.get('test', '-')} |"
        )

    # 最终指标表
    lines += [
        "",
        "## 最终指标（最后一轮）",
        "",
        "| 任务 | precision | recall | mAP50 | mAP50-95 |",
        "|------|-----------|--------|-------|----------|",
    ]
    for task in tasks:
        cols = data[task]
        def last(key):
            v = cols.get(key)
            return f"{v[-1]:.4f}" if v else "-"
        lines.append(
            f"| {task} | {last('metrics/precision(B)')} | "
            f"{last('metrics/recall(B)')} | {last('metrics/mAP50(B)')} | "
            f"{last('metrics/mAP50-95(B)')} |"
        )

    # 趋势分析
    trends = {task: analyze_trend(data[task]) for task in tasks}
    lines += [
        "",
        "## 曲线趋势分析",
        "",
        "| 任务 | epochs | 最后20轮 mAP50 均值 | 最后20轮 mAP50 最高 | box_loss | 收敛状态 |",
        "|------|--------|---------------------|---------------------|----------|----------|",
    ]
    for task in tasks:
        t = trends[task]
        status = "✅ 稳定收敛" if t["stable"] else "⚠️ 未收敛"

        def fmt3(v):
            return f"{v:.3f}" if v is not None else "-"

        lines.append(
            f"| {task} | {t['epochs']} | {fmt3(t['last20_mean'])} | "
            f"{fmt3(t['last20_max'])} | {fmt3(t['box_loss'])} | {status} |"
        )

    # 总结与建议
    lines += [
        "",
        "## 总结与建议",
        "",
    ]
    for task in tasks:
        t = trends[task]
        rec = make_recommendations(task, t)
        lines.append(f"- **{task}**: {rec}")

    lines += [
        "",
        "## 内容说明",
        "",
        f"- `charts/` — 各指标随 epoch 变化的曲线图"
        + ("（多任务叠加对比）" if len(tasks) > 1 else ""),
        f"- `artifacts/` — 训练产物图（PR/F1 曲线、混淆矩阵、批次样本等）",
        f"- `data/` — results.csv 副本",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
