# YOLO 训练任务生成 — 实现计划

## 背景

数据集 `0002_20270812` 已创建完成（步骤4）。根据 `raw_spec.txt`，接下来进入步骤5：创建训练任务。训练任务包含数据集、训练脚本、运维脚本（启动/查看日志/停止），最终打包为 `.tar.gz` 上传到训练服务器使用。

## 参考格式分析

当前 `05_train/0001-2026-08-11-yolov11n/` 结构：

```
0001-2026-08-11-yolov11n/
├── 0002_20270812.tar.gz          # 打包的数据集
└── script/
    ├── train_rv1106_bz.sh         # 训练核心脚本
    └── train_rv1106_bz_execute.sh # nohup 后台启动
```

**train_rv1106_bz.sh** — 接收 `ver`（data.yaml路径）和 `epochs`，激活 conda 环境，执行 `yolo detect train`：
```bash
#!/bin/bash
ver=$1
epochs=$2
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rv1106-ultralytics
cd /root/ultralytics-8.3.39-rv1106 && \
yolo detect train data=${ver} model=yolo11n.pt epochs=${epochs} workers=10 batch=24 imgsz=640
```

**train_rv1106_bz_execute.sh** — 接收 `modelVersionId` 和 `epochs`，用 nohup 后台启动训练，输出日志到 `/root/bash/rv1106-ultralytics/logs/`：
```bash
#/bin/bash
modelVersionId=$1
epochs=$2
TIME=`/bin/date +%Y%m%d%H%M%S`
nohup ./train_rv1106_bz.sh $modelVersionId $epochs >> /root/bash/rv1106-ultralytics/logs/train_${modelVersionId}_${TIME}.log 2>&1 &
```

## 新增脚本设计

需要新增三个运维脚本，所有脚本顶部用变量定义路径，方便不同环境调整。

### start.sh — 启动训练
```bash
#!/bin/bash
# ====== 可配置变量 ======
DATASET_DIR="./0002_20270812"
EPOCHS=200
# ========================

DATA_YAML="${DATASET_DIR}/data.yaml"
./script/train_rv1106_bz_execute.sh ${DATA_YAML} ${EPOCHS}
```
- 默认 epochs=200，解压后直接 `./start.sh` 运行
- 也支持 `./start.sh 300` 手动指定 epochs

### hdlog.sh — 查看日志
```bash
#!/bin/bash
# ====== 可配置变量 ======
LOGS_DIR="/root/bash/rv1106-ultralytics/logs"
# ========================

echo "=== 日志目录 ==="
ls -al ${LOGS_DIR}
echo ""
echo "=== 最新日志 ==="
LATEST=$(ls -t ${LOGS_DIR}/train_*.log 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "实时查看: ${LATEST}"
    tail -n 200 -f "${LATEST}"
else
    echo "未找到训练日志"
fi
```
- 列出日志目录
- 自动找到最新日志并用 `tail -f` 实时查看

### stop.sh — 停止训练
```bash
#!/bin/bash
echo "=== 查找训练进程 ==="
ps aux | grep "yolo detect train" | grep -v grep

PIDS=$(ps aux | grep "yolo detect train" | grep -v grep | awk '{print $2}')
if [ -z "$PIDS" ]; then
    echo "未找到运行中的训练进程"
    exit 0
fi

echo ""
echo "找到进程 PID: $PIDS"
echo "正在发送 SIGTERM (15)..."
for pid in $PIDS; do
    kill -15 $pid
    echo "  已终止 PID: $pid"
done
echo "训练已停止"
```
- 自动查找 yolo 训练进程
- 用 `kill -15`（SIGTERM，优雅退出）终止

## Python 脚本设计：`scripts/create_train_task.py`

### CLI 参数

```bash
python scripts/create_train_task.py \
  --dataset Root/04_dataset/0002_20270812 \    # 数据集路径（目录或 tar.gz）
  --name 0002-20260812-yolov11n \               # 训练任务名称
  --train-dir Root/05_train \                   # 训练任务输出根目录（默认）
  --model yolo11n \                             # 模型名
  --epochs 200 \                                # 默认训练轮数
  --workers 10 \                                # 数据加载线程数
  --batch 24 \                                  # 批次大小
  --imgsz 640 \                                 # 输入图片尺寸
  --output-tar \                                # 是否也输出 .tar.gz（默认是）
  --overwrite                                   # 覆盖已存在任务
```

### 处理流程

1. **创建任务目录**: `{train-dir}/{name}/script/`
2. **复制/打包数据集**: 
   - 如果 `--dataset` 是目录 → 先打包为 `{dataset_name}.tar.gz`，放到任务目录下
   - 如果 `--dataset` 已经是 tar.gz → 直接复制
   - 同时解压一份数据集到任务目录下（方便直接运行 `start.sh`）
3. **生成训练脚本** (`script/train_rv1106_bz.sh`): 
   - 变量: `CONDA_PATH`, `CONDA_ENV`, `ULTRALYTICS_PATH`, `MODEL`, `WORKERS`, `BATCH`, `IMGSZ`
   - 核心逻辑与参考一致
4. **生成启动脚本** (`script/train_rv1106_bz_execute.sh`):
   - 变量: `LOGS_DIR`
   - nohup 后台启动
5. **生成运维脚本** (`start.sh`, `hdlog.sh`, `stop.sh`):
   - 所有服务器路径用变量定义在脚本顶部
6. **生成 README.txt**: 快速使用说明
7. **最终打包**: `cd {train-dir} && tar czf {name}.tar.gz {name}/`

### 输出结构

```
05_train/0002-20260812-yolov11n/
├── 0002_20270812/                    # 解压后的数据集（方便直接使用）
├── 0002_20270812.tar.gz              # 数据集归档
├── start.sh                          # 启动训练
├── hdlog.sh                          # 查看日志
├── stop.sh                           # 停止训练
├── README.txt                        # 使用说明
└── script/
    ├── train_rv1106_bz.sh            # 训练核心
    └── train_rv1106_bz_execute.sh    # nohup 启动
```

最终打包:
```
05_train/0002-20260812-yolov11n.tar.gz
```

## 使用流程

```bash
# 1. 生成训练任务
python scripts/create_train_task.py \
  --dataset Root/04_dataset/0002_20270812 \
  --name 0002-20260812-yolov11n \
  --epochs 200

# 2. 上传到训练服务器
scp Root/05_train/0002-20260812-yolov11n.tar.gz user@server:/path/

# 3. 在服务器上解压运行
tar xzf 0002-20260812-yolov11n.tar.gz
cd 0002-20260812-yolov11n
./start.sh          # 启动训练
./hdlog.sh          # 查看日志
./stop.sh           # 停止训练
```

## 涉及文件

| 文件 | 操作 |
|------|------|
| `scripts/create_train_task.py` | **新建** — 训练任务生成脚本 |
| `Root/05_train/0002-20260812-yolov11n/` | **新建** — 训练任务目录 |
| `Root/05_train/0002-20260812-yolov11n.tar.gz` | **新建** — 最终打包 |
| `spec/create_train_task_plan.md` | **新建** — 本计划文档 |

## 验证方法

1. 运行 `python scripts/create_train_task.py --dataset Root/04_dataset/0002_20270812 --name 0002-20260812-yolov11n`
2. 检查输出目录结构完整
3. 检查所有脚本内容正确、变量可配置
4. 检查 tar.gz 打包正确: `tar tzf Root/05_train/0002-20260812-yolov11n.tar.gz`
5. 在临时目录解压测试: `start.sh` 参数传递正确
