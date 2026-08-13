# CLAUDE.md

本文件为 Claude Code 在此项目中的工作指引。

## 项目说明

YOLO 目标识别训练流水线项目，目录结构：

```
.server.conf.local     # 本地服务器连接配置（gitignored，模板: Root/00_config/server.conf.example）
Root/
├── 00_config/          # 全局配置（classes.txt、server.conf.example）
├── 01_raw/             # ① 原始视频（人工）
├── 02_label/           # ② 标注（人工）
├── 03_label_check/     # ③ 标注检查（人工）
├── 04_dataset/         # ④ 数据集（scripts/create_dataset.py）
├── 05_train/           # ⑥ 训练任务（scripts/create_train_task.py）
├── 06_data_analysis/   # ⑨ 结果分析（scripts/analyze_results.py）
└── 06_val/             # 验证（预留）
model/                  # 公共模型（yolo26n.pt，不入库）
scripts/                # 工具脚本
spec/                   # 文档
```

## 训练流水线（① → ⑨）

```
① 素材采集 ──→ ② 标注 ──→ ③ 标注检查 ──→ ④ 创建数据集 ──→ ⑤ 可视化抽查 ──→ ⑥ 创建训练任务 ──→ ⑦ 服务器训练 ──→ ⑧ 结果打包回传 ──→ ⑨ 结果分析
   (人工)       (人工)       (人工)          (脚本)          (脚本,可选)      (脚本)           (服务器操作)       (服务器脚本)       (脚本)
```

### ① 素材采集（人工）

拍摄人员按场景存放视频：`01_raw/scene_XXXX/*.mp4`

### ② 标注（人工）

逐帧标注，每帧三个文件（txt 为空文件 = 无目标）：

```
02_label/scene_XXXX/视频名/frame_XXXXX.jpg    # 帧图片
02_label/scene_XXXX/视频名/frame_XXXXX.json   # LabelMe 标注
02_label/scene_XXXX/视频名/frame_XXXXX.txt    # YOLO 标注
```

### ③ 标注检查（人工）

检查人员复制确认后的标注到 `03_label_check/scene_XXXX/视频名/`。

### ④ 创建数据集（脚本）

```bash
# 单场景
python3 scripts/create_dataset.py -s Root/03_label_check/scene_0001 -n 0004_20260813

# 多场景合并（-s 接多个）+ 自定义拆分比例 / 正空比例 / 随机种子
python3 scripts/create_dataset.py \
  -s Root/03_label_check/scene_0001 Root/03_label_check/scene_0002 \
  -n 0005_20260813 --split 7 2 1 --pos-neg-ratio 9 1 --seed 42
```

- `-s/--source` 必填（可多个），`-n/--name` 必填
- `--split` 默认 `7 2 1`（train:val:test），`--pos-neg-ratio` 默认 `9 1`（正:空），`--rounding` 默认 `floor`
- `--dry-run` 仅预览分配方案
- 输出 `04_dataset/0004_20260813/`：`data.yaml` + `dataset_stats.txt` + `images/{train,val,test}` + `labels/{train,val,test}`

### ⑤ 标注可视化抽查（脚本，可选）

```bash
python3 scripts/draw_bbox.py \
  -s Root/04_dataset/0004_20260813 \
  -o output/vis \
  -c Root/00_config/classes.txt
```

输出 `output/vis/{train,val}/` 带框标注图片，训练前人工检查标注质量。

### ⑥ 创建训练任务（脚本）

```bash
python3 scripts/create_train_task.py \
  --dataset Root/04_dataset/0004_20260813 \
  --name 0004-20260813-yolov11s \
  --model yolo11s --epochs 200
```

- `--dataset`、`--name` 必填；`--model` 默认 `yolo11n`，`--epochs` 默认 `200`
- `--conda-env` 默认 `rv1106-ultralytics`，`--ultralytics-path` 默认 `/root/ultralytics-8.3.39-rv1106`
- `--no-export-onnx` 可关闭 pack_result.sh 的自动 ONNX 导出（默认开启）
- 输出 `05_train/0004-20260813-yolov11s.tar.gz`，内含 `dataset/`、`logs/`、`start.sh`、`hdlog.sh`、`stop.sh`、`pack_result.sh`、`export_onnx.sh`、`script/`
- 自动打包两份 `yolo26n.pt`（任务根目录 + `dataset/` 内）——训练进程 CWD 是 dataset/，ultralytics 只在 CWD 查找，这份才能阻止 AMP 检查时联网下载

### ⑦ 服务器训练（服务器操作）

上传解压任务包后：

```bash
./start.sh          # 启动训练（默认 epochs）
./start.sh 1        # 指定 epochs（快速测试）
./hdlog.sh          # 实时查看日志
./stop.sh           # 停止训练（kill -15 优雅退出）
./export_onnx.sh    # best.pt → best.onnx 导出（已存在则跳过，不覆盖）
```

### ⑧ 结果打包回传（服务器脚本）

```bash
./pack_result.sh    # → result_时间戳.tar.gz（时间戳命名，不覆盖旧包；打包前自动导出 ONNX）
```

打包内容：`train/`（weights/best.pt、**weights/best.onnx**、results.csv、产物图）+ `logs/`。
下载最新 `result_*.tar.gz` 回本地 `05_train/任务名/`。

### ⑨ 结果分析（脚本）

```bash
# 分析单个任务（依赖: pip install matplotlib）
python3 scripts/analyze_results.py --tasks 0005-20260813-yolov11n

# 对比多个任务（用户明确提出对比时才使用）
python3 scripts/analyze_results.py --tasks 0005-20260813-yolov11n 0004-20260813-yolov11s
```

输出 `06_data_analysis/{自动命名}/`：`summary.md`（指标表+收敛分析）+ `charts/`（指标曲线）+ `artifacts/`（产物图）+ `data/`（CSV 副本）。

### 服务器状态查看（跨平台）

```bash
# 依赖: pip install paramiko；读仓库根目录 .server.conf.local
python3 scripts/check_server_status.py              # 读默认配置
python3 scripts/check_server_status.py --host 2.3.4.5   # 临时换服务器
```

查看内容：训练进程 / GPU / 最新日志 / 训练指标 / 磁盘空间。

## 铁律：训练服务器文件操作规则

运行在训练服务器里的脚本和代码（start.sh / hdlog.sh / stop.sh / pack_result.sh / export_onnx.sh / train_rv1106_bz.sh / train_rv1106_bz_execute.sh 以及未来所有部署到服务器的脚本）必须遵循：

1. **当前任务文件夹内**：脚本/代码可以创建文件、移动文件，**不可以删除文件**。
2. **当前任务文件夹外**：绝对禁止所有脚本/代码修改、删除、移动任何任务文件。

### 适用解读

- "当前任务文件夹" = 脚本所属的训练任务目录（如 `0004-20260813-yolov11s/`）
- 禁止删除 = 不使用 `rm`、不覆盖写入已有文件（覆盖写入视为先删除后创建）
- 文件夹外只读 = 服务器上的 conda、ultralytics、系统目录一律只读访问
- 需要清理的场景（如重跑训练）：必须由操作人员手动执行，脚本不得自动删除
- 脚本写日志、写训练结果等"创建"行为不受限制（ultralytics 自身行为除外）

## 训练结果对比规则

**只有用户明确提出对比时，才对比两次训练任务的结果**；用户没有要求对比时，只报告当前任务的结果，不主动与其他任务对比。

## Git 提交规则

**提交 git 前必须先征得用户确认**，不得自行 commit/push。确认时需说明：提交的文件清单、变更摘要。用户确认后才能执行提交。

## 脚本编写约定

- 本地工具脚本放 `scripts/`，Python 优先
- 服务器脚本生成器为 `scripts/create_train_task.py`，修改服务器脚本模板时必须同步检查上述铁律
- 脚本顶部用 `# ====== 可配置变量 ======` 区声明可修改项
- 所有脚本必须在 Mac / Linux / Windows (Git Bash) 上语法兼容（避免 GNU-only 参数）
