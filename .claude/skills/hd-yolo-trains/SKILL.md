---
name: hd-yolo-trains
description: YOLO 训练流水线全流程操作指引——创建数据集、打包训练任务、服务器训练运维、结果打包回传、best.pt→best.onnx 导出、结果分析、服务器状态查看。当用户提到训练、数据集、服务器、结果分析、onnx 导出或打包回传时使用。
---

# hd-yolo-trains 训练流水线操作指引

## 流程总览

```
① 素材采集 ──→ ② 标注 ──→ ③ 标注检查 ──→ ④ 创建数据集 ──→ ⑤ 可视化抽查 ──→ ⑥ 创建训练任务 ──→ ⑦ 服务器训练 ──→ ⑧ 结果打包回传 ──→ ⑨ 结果分析
   (人工)       (人工)       (人工)          (脚本)          (脚本,可选)      (脚本)           (服务器操作)       (服务器脚本)       (脚本)
```

目录约定：

```
.server.conf.local     # 本地服务器连接配置（含密码，gitignored，模板: Root/00_config/server.conf.example）
Root/
├── 00_config/          # classes.txt（类别单一数据源）、server.conf.example
├── 01_raw/             # ① 原始视频: 01_raw/scene_XXXX/*.mp4
├── 02_label/           # ② 标注: scene_XXXX/视频名/frame_XXXXX.{jpg,json,txt}（空 txt = 无目标）
├── 03_label_check/     # ③ 标注检查（检查确认后复制到这里）
├── 04_dataset/         # ④ 数据集
├── 05_train/           # ⑥ 训练任务（含回传的 result_*.tar.gz）
├── 06_data_analysis/   # ⑨ 结果分析
└── 06_val/             # （预留）验证
model/                  # yolo26n.pt 公共模型（不入库）
scripts/                # 本地工具脚本
spec/                   # 文档（pipeline_workflow.md 为全流程详细文档）
```

## 铁律：训练服务器文件操作规则（最高优先级）

部署到服务器的脚本（start.sh / hdlog.sh / stop.sh / pack_result.sh / export_onnx.sh / train_rv1106_bz.sh / train_rv1106_bz_execute.sh 及未来所有服务器脚本）必须遵循：

1. **当前任务文件夹内**：可以创建文件、移动文件，**不可以删除文件**。
2. **当前任务文件夹外**：绝对禁止修改、删除、移动任何任务文件。

适用解读：
- "当前任务文件夹" = 脚本所属的训练任务目录（如 `0004-20260813-yolov11s/`）
- 禁止删除 = 不用 `rm`、不覆盖写入已有文件（覆盖视为先删除后创建）
- 文件夹外只读 = conda、ultralytics、系统目录一律只读
- 需要清理（如重跑训练）：必须由操作人员手动执行，脚本不得自动删除
- 脚本写日志、写训练结果等"创建"行为不受限制

**修改服务器脚本模板时（scripts/create_train_task.py 中的 generate_* 函数），逐条对照以上铁律检查。**

## 各步骤操作

### ④ 创建数据集

```bash
python3 scripts/create_dataset.py -s Root/03_label_check/scene_0001 -n 0004_20260813
# 多场景: -s 接多个目录；--split 7 2 1（train:val:test）；--pos-neg-ratio 9 1（正:空）；--seed 42；--dry-run 仅预览
```

输出 `04_dataset/0004_20260813/`：data.yaml + dataset_stats.txt + images/labels 按 train/val/test。

### ⑤ 标注可视化抽查（可选）

```bash
python3 scripts/draw_bbox.py -s Root/04_dataset/0004_20260813 -o output/vis -c Root/00_config/classes.txt
```

### ⑥ 创建训练任务

```bash
python3 scripts/create_train_task.py \
  --dataset Root/04_dataset/0004_20260813 \
  --name 0004-20260813-yolov11s \
  --model yolo11s --epochs 200
```

- `--dataset`、`--name` 必填；`--model` 默认 yolo11n；`--epochs` 默认 200
- `--conda-env` 默认 rv1106-ultralytics；`--ultralytics-path` 默认 /root/ultralytics-8.3.39-rv1106
- `--no-tar` 不打包；`--no-export-onnx` 关闭自动 ONNX 导出；`--overwrite` 覆盖已存在任务目录
- 自动打包两份 yolo26n.pt（任务根目录 + dataset/ 内）——训练进程 CWD 是 dataset/，ultralytics 只在 CWD 查找，才能阻止 AMP 检查联网下载
- 输出 `05_train/0004-20260813-yolov11s.tar.gz`，内含 dataset/、logs/、start.sh、hdlog.sh、stop.sh、pack_result.sh、export_onnx.sh、script/

### ⑦ 服务器训练

上传解压任务包后：

```bash
./start.sh          # 启动训练（默认 epochs）
./start.sh 1        # 指定 epochs（快速测试）
./hdlog.sh          # 实时查看日志（自动 tail 最新）
./stop.sh           # 停止训练（kill -15 优雅退出）
./export_onnx.sh    # best.pt → best.onnx 导出（已存在则跳过，不覆盖）
```

### ⑧ 结果打包回传

```bash
./pack_result.sh    # → result_时间戳.tar.gz（时间戳命名，不覆盖旧包；打包前自动导出 ONNX）
```

打包内容：`train/`（weights/best.pt、weights/best.onnx、results.csv、产物图）+ `logs/`。
下载最新 `result_*.tar.gz` 回本地 `05_train/任务名/`。
导出失败不阻断打包（仅警告）；best.onnx 已存在则跳过（铁律：不覆盖）。

### ⑨ 结果分析

```bash
# 依赖: pip install matplotlib
python3 scripts/analyze_results.py --tasks 0005-20260813-yolov11n
# 多任务对比仅在用户明确提出对比时使用:
python3 scripts/analyze_results.py --tasks A B
```

输出 `06_data_analysis/{自动命名}/`：summary.md（指标表+收敛分析）+ charts/（指标曲线）+ artifacts/（产物图）+ data/（CSV 副本）。

### 服务器状态查看

```bash
# 依赖: pip install paramiko；读仓库根目录 .server.conf.local
python3 scripts/check_server_status.py              # 默认配置
python3 scripts/check_server_status.py --host 2.3.4.5   # 临时换服务器
```

查看：训练进程 / GPU / 最新日志 / 训练指标 / 磁盘空间。
注意：该脚本只显示"最新"任务的状态；查指定任务需 SSH 定向查询该任务目录（进程 grep 任务名、tail 其 logs/train_*.log、tail 其 */runs/detect/train*/results.csv）。

## 必须遵守的规则

1. **结果对比**：只在用户明确提出对比时，才对比多个训练任务的结果；否则只报告当前任务。
2. **Git**：提交/推送前必须先征得用户确认，说明文件清单和变更摘要。
3. **脚本约定**：本地工具脚本放 scripts/（Python 优先）；服务器脚本模板在 create_train_task.py；脚本顶部用 `# ====== 可配置变量 ======` 区声明可修改项；本地脚本须 Mac/Linux/Windows (Git Bash) 兼容（避免 GNU-only 参数）。
4. **敏感信息**：服务器密码只存 .server.conf.local（gitignored），绝不写入脚本/文档/提交。

## 常见任务速查

| 需求 | 做法 |
|------|------|
| 查最新训练进度 | `python3 scripts/check_server_status.py` |
| 查指定任务进度 | SSH 到服务器定向查该任务目录（日志 tail + results.csv tail + 进程） |
| 分析训练结果 | `python3 scripts/analyze_results.py --tasks <任务名>` |
| 导出 onnx | 服务器任务目录 `./export_onnx.sh`（或打包时自动完成） |
| 新建训练任务 | ④ → ⑥ 流程，自动打包 tar.gz |
| 修改服务器脚本 | 改 create_train_task.py 对应 generate_* 模板，对照铁律逐条检查 |
