#!/bin/bash
# ============================================================
# 训练服务器状态查看脚本（本地 Mac 运行）
#
# 查看内容:
#   1. 训练进程状态
#   2. GPU 状态 (nvidia-smi)
#   3. 最新训练日志尾部
#   4. 最新训练指标 (results.csv)
#   5. 磁盘空间
#
# 依赖: sshpass (brew install sshpass) 或手动输入密码
# ============================================================

# ====== 配置（按需修改） ======
SERVER_USER="root"
SERVER_IP=""                      # ← 填服务器 IP
SERVER_PORT=22
TASK_ROOT="/root/bash/rv1106-ultralytics/data/dataset"   # 服务器上训练任务目录
# ==============================

if [ -z "$SERVER_IP" ]; then
    echo "[错误] 请先在脚本顶部填写 SERVER_IP"
    exit 1
fi

# 读取密码（不显示）
if command -v sshpass >/dev/null 2>&1; then
    read -sp "服务器密码: " PASS
    echo ""
    SSH_CMD="sshpass -p ${PASS} ssh -o StrictHostKeyChecking=no -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP}"
else
    echo "[提示] 未安装 sshpass，请安装: brew install sshpass"
    echo "[提示] 或将密码写入脚本顶部 PASSWORD 变量（不推荐）"
    SSH_CMD="ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP}"
fi

# 远程执行状态检查
${SSH_CMD} TASK_ROOT="${TASK_ROOT}" 'bash -s' << 'REMOTE_EOF'
echo "============================================================"
echo "  训练服务器状态"
echo "  $(hostname) @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo ""
echo "========== 1. 训练进程 =========="
ps aux | grep "yolo detect train" | grep -v grep && echo "" && \
echo "运行时长:" && ps -o etime= -p $(pgrep -f "yolo detect train" | head -1) 2>/dev/null || \
echo "  ❌ 无训练进程运行"

echo ""
echo "========== 2. GPU 状态 =========="
nvidia-smi

echo ""
echo "========== 3. 最新训练日志 =========="
LATEST_LOG=$(ls -t ${TASK_ROOT}/*/logs/train_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  日志: $LATEST_LOG"
    echo "  --- 最后 15 行 ---"
    tail -n 15 "$LATEST_LOG" | sed 's/^/  | /'
else
    echo "  未找到训练日志"
fi

echo ""
echo "========== 4. 训练指标 =========="
LATEST_CSV=$(ls -t ${TASK_ROOT}/*/dataset/runs/detect/train*/results.csv 2>/dev/null | head -1)
if [ -n "$LATEST_CSV" ]; then
    echo "  结果: $LATEST_CSV"
    echo "  --- 最新 epoch ---"
    tail -n 1 "$LATEST_CSV" | tr ',' '\n' | sed 's/^/  | /'
else
    echo "  未找到 results.csv（训练可能刚开始或未完成第 1 个 epoch）"
fi

echo ""
echo "========== 5. 磁盘空间 =========="
df -h / /data 2>/dev/null | grep -v "^Filesystem" | sed 's/^/  /'

echo ""
echo "============================================================"
REMOTE_EOF
