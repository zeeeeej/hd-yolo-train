# YOLO 训练全流程

## 流程总览

```
① 素材采集 ──→ ② 标注 ──→ ③ 标注检查 ──→ ④ 创建数据集 ──→ ⑤ 可视化抽查 ──→ ⑥ 创建训练任务 ──→ ⑦ 服务器训练 ──→ ⑧ 结果打包回传 ──→ ⑨ 结果分析
   (人工)       (人工)       (人工)          (脚本)          (脚本,可选)      (脚本)           (服务器操作)       (服务器脚本)       (脚本)
```

---

## ① 素材采集（人工）

拍摄人员按场景存放视频：

```
01_raw/scene_XXXX/*.mp4
```

## ② 标注（人工）

标注人员逐帧标注，每帧三个文件：

```
02_label/scene_XXXX/视频名/frame_XXXXX.jpg    # 帧图片
02_label/scene_XXXX/视频名/frame_XXXXX.json   # LabelMe 标注
02_label/scene_XXXX/视频名/frame_XXXXX.txt    # YOLO 标注（空文件 = 无目标）
```

## ③ 标注检查（人工）

检查人员复制确认后的标注：

```
03_label_check/scene_XXXX/视频名/
```

## ④ 创建数据集（脚本）

```bash
# 单场景
python3 scripts/create_dataset.py -s Root/03_label_check/scene_0001 -n 0004_20260813

# 多场景合并
python3 scripts/create_dataset.py \
  -s Root/03_label_check/scene_0001 Root/03_label_check/scene_0002 \
  -n 0005_20260813

# 自定义拆分比例 / 正空比例 / 随机种子
python3 scripts/create_dataset.py \
  -s Root/03_label_check/scene_0001 \
  -n 0005_20260813 \
  --split 7 2 1 \
  --pos-neg-ratio 9 1 \
  --seed 42
```

参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-s/--source` | 必填 | 数据源，可多个（场景目录或任意文件夹） |
| `-n/--name` | 必填 | 数据集名称（如 0004_20260813） |
| `--split` | `7 2 1` | train/val/test 比例 |
| `--pos-neg-ratio` | `9 1` | 正样本/空样本比例（严格控制） |
| `--rounding` | `floor` | 空样本取整方式（floor=严格/nearest=四舍五入） |
| `--seed` | `42` | 随机种子（可复现） |
| `--dry-run` | 关闭 | 仅预览分配方案 |

输出：`04_dataset/0004_20260813/`

```
0004_20260813/
├── data.yaml            # YOLO 配置（path: . 相对路径）
├── dataset_stats.txt    # 分配统计
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

## ⑤ 标注可视化抽查（脚本，可选）

训练前人工检查标注质量：

```bash
python3 scripts/draw_bbox.py \
  -s Root/04_dataset/0004_20260813 \
  -o output/vis \
  -c Root/00_config/classes.txt
```

输出：`output/vis/{train,val}/` 带框标注图片，可直观查看标注是否正确。

## ⑥ 创建训练任务（脚本）

```bash
python3 scripts/create_train_task.py \
  --dataset Root/04_dataset/0004_20260813 \
  --name 0004-20260813-yolov11s \
  --model yolo11s \
  --epochs 200
```

参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集路径（目录或 tar.gz） |
| `--name` | 必填 | 训练任务名（如 0004-20260813-yolov11s） |
| `--model` | `yolo11n` | 模型名（yolo11n/yolo11s/...） |
| `--epochs` | `200` | 默认训练轮数 |
| `--workers` | `10` | 数据加载线程数 |
| `--batch` | `24` | 批次大小 |
| `--imgsz` | `640` | 输入图片尺寸 |
| `--conda-env` | `rv1106-ultralytics` | 服务器 conda 环境 |
| `--ultralytics-path` | `/root/ultralytics-8.3.39-rv1106` | 服务器 ultralytics 路径 |

输出：`05_train/0004-20260813-yolov11s.tar.gz`

```
0004-20260813-yolov11s/
├── dataset/                  # 数据集（统一命名）
├── logs/                     # 训练日志（运行时生成）
├── start.sh                  # 启动训练
├── hdlog.sh                  # 实时日志
├── stop.sh                   # 停止训练
├── pack_result.sh            # 打包训练结果
├── README.txt
└── script/
    ├── train_rv1106_bz.sh         # 训练核心（可配置变量在顶部）
    └── train_rv1106_bz_execute.sh # nohup 后台启动
```

## ⑦ 服务器训练

上传任务包到服务器，解压运行：

```bash
tar xzf 0004-20260813-yolov11s.tar.gz
cd 0004-20260813-yolov11s

./start.sh          # 默认 epochs 启动
./start.sh 1        # 指定 epochs（快速测试）
./hdlog.sh          # 实时查看日志（自动 tail 最新）
./stop.sh           # 停止训练（kill -15 优雅退出）
```

## ⑧ 结果打包回传（服务器）

```bash
./pack_result.sh    # → result_20260813_153022.tar.gz（时间戳命名，不覆盖旧包）
```

打包内容：
```
result_时间戳.tar.gz
├── train/                # 训练结果（weights/best.pt、results.csv 等）
└── logs/                 # 训练日志
```

下载最新的 `result_*.tar.gz` 回本地归档分析。注意服务器脚本遵循铁律（见 CLAUDE.md）：不删除任何文件，重复打包只会新增时间戳包。

---

## ⑨ 结果分析（脚本）

```bash
# 分析单个任务
python3 scripts/analyze_results.py --tasks 0005-20260813-yolov11n

# 对比多个任务（明确提出对比才使用）
python3 scripts/analyze_results.py --tasks 0005-20260813-yolov11n 0004-20260813-yolov11s
```

输出到 `06_data_analysis/{自动命名}/`：

```
06_data_analysis/0005_vs_0004_20260813/
├── summary.md             # 汇总报告（任务概览 + 数据集场景 + 最终指标表）
├── charts/                # 各指标随 epoch 变化的曲线图（多任务叠加对比）
├── artifacts/{task}/      # 训练产物图（PR/F1 曲线、混淆矩阵、批次样本等）
└── data/{task}_results.csv
```

依赖: `pip install matplotlib`

---

## 工具链总览

| 脚本 | 用途 | 运行位置 |
|------|------|----------|
| `scripts/create_dataset.py` | 数据集创建（多场景合并/比例拆分/JSON→YOLO） | 本地/Docker |
| `scripts/draw_bbox.py` | 标注可视化抽查 | 本地/Docker |
| `scripts/create_train_task.py` | 训练任务打包（脚本+dataset+yolo26n.pt+tar.gz） | 本地/Docker |
| `scripts/analyze_results.py` | 训练结果分析（曲线图/产物图/汇总报告） | 本地/Docker |
| `start.sh` | 启动训练 | 训练服务器 |
| `hdlog.sh` | 实时日志 | 训练服务器 |
| `stop.sh` | 停止训练 | 训练服务器 |
| `pack_result.sh` | 训练结果+日志打包 | 训练服务器 |
| `scripts/check_server_status.py` | 查看服务器训练状态（进程/GPU/日志/指标/磁盘），跨平台 | 本地 Mac/Windows/Docker |

## 公共模型 model/

`model/yolo26n.pt` 是公共模型（服务器 conda 版 ultralytics 8.4.x 的默认模型，AMP 检查会用到）。
创建训练任务时自动打包两份：
- 任务根目录（与 script/ 平级）
- `dataset/` 内 —— 训练进程 CWD 是 dataset/，ultralytics 只在 CWD 查找模型，这份才能阻止 AMP 检查时联网下载

## 服务器状态查看（跨平台）

```bash
# 首次配置: 复制 Root/00_config/server.conf.example 为仓库根目录 .server.conf.local 并填写服务器 IP
# 依赖: pip install paramiko

python3 scripts/check_server_status.py              # 读默认配置
python3 scripts/check_server_status.py --host 2.3.4.5   # 临时换服务器
```

查看内容：训练进程 / GPU (nvidia-smi) / 最新日志 / 训练指标 (results.csv) / 磁盘空间。

## 目录约定

```
ai-yolo-train/
├── .server.conf.local      # 本地服务器连接配置（含密码，gitignored）
├── model/                  # 公共模型（yolo26n.pt）
├── scripts/                # 本地/Docker 工具脚本
├── spec/                   # 文档
└── Root/
    ├── 00_config/          # 全局配置（classes.txt、server.conf.example）
    ├── 01_raw/             # ① 原始视频
    ├── 02_label/           # ② 标注
    ├── 03_label_check/     # ③ 标注检查
    ├── 04_dataset/         # ④ 数据集
    ├── 05_train/           # ⑥ 训练任务（含回传的 result_*.tar.gz）
    ├── 06_data_analysis/   # ⑨ 结果分析
    └── 06_val/             # （预留）验证
```
