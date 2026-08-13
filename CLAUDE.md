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
├── 05_train/           # ⑤ 训练任务（scripts/create_train_task.py）
└── 06_val/             # ⑥ 验证（预留）
scripts/                # 工具脚本
spec/                   # 文档
```

完整流程见 `spec/pipeline_workflow.md`。

## 铁律：训练服务器文件操作规则

运行在训练服务器里的脚本和代码（start.sh / hdlog.sh / stop.sh / pack_result.sh / train_rv1106_bz.sh / train_rv1106_bz_execute.sh 以及未来所有部署到服务器的脚本）必须遵循：

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
