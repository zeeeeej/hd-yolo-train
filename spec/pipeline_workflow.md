# YOLO 训练全流程

## 流程总览

```
① 素材采集 ──→ ② 标注 ──→ ③ 标注检查 ──→ ④ 创建数据集 ──→ ⑤ 可视化抽查 ──→ ⑥ 创建训练任务 ──→ ⑦ 服务器训练 ──→ ⑧ 结果打包回传
   (人工)       (人工)       (人工)          (脚本)          (脚本,可选)      (脚本)           (服务器操作)       (服务器脚本)
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
./pack_result.sh    # → result.tar.gz
```

打包内容：
```
result.tar.gz
├── train/                # 训练结果（weights/best.pt、results.csv 等）
└── logs/                 # 训练日志
```

下载 `result.tar.gz` 回本地归档分析。

---

## 工具链总览

| 脚本 | 用途 | 运行位置 |
|------|------|----------|
| `scripts/create_dataset.py` | 数据集创建（多场景合并/比例拆分/JSON→YOLO） | 本地/Docker |
| `scripts/draw_bbox.py` | 标注可视化抽查 | 本地/Docker |
| `scripts/create_train_task.py` | 训练任务打包（脚本+dataset+tar.gz） | 本地/Docker |
| `start.sh` | 启动训练 | 训练服务器 |
| `hdlog.sh` | 实时日志 | 训练服务器 |
| `stop.sh` | 停止训练 | 训练服务器 |
| `pack_result.sh` | 训练结果+日志打包 | 训练服务器 |
| `scripts/check_server_status.py` | 查看服务器训练状态（进程/GPU/日志/指标/磁盘），跨平台 | 本地 Mac/Windows/Docker |

## 服务器状态查看（跨平台）

```bash
# 首次配置: 编辑 Root/00_config/server.conf 填写服务器 IP
# 依赖: pip install paramiko

python3 scripts/check_server_status.py              # 读默认配置
python3 scripts/check_server_status.py --host 2.3.4.5   # 临时换服务器
```

查看内容：训练进程 / GPU (nvidia-smi) / 最新日志 / 训练指标 (results.csv) / 磁盘空间。

## 目录约定

```
ai-yolo-train/
├── scripts/                # 本地/Docker 工具脚本
├── spec/                   # 文档
└── Root/
    ├── 00_config/          # 全局配置（classes.txt 单一数据源）
    ├── 01_raw/             # ① 原始视频
    ├── 02_label/           # ② 标注
    ├── 03_label_check/     # ③ 标注检查
    ├── 04_dataset/         # ④ 数据集
    ├── 05_train/           # ⑥ 训练任务
    └── 06_val/             # （预留）验证
```
