#!/usr/bin/env python3
"""
训练服务器状态查看脚本（跨平台：Windows / macOS / Linux）

查看内容:
  1. 训练进程状态
  2. GPU 状态 (nvidia-smi)
  3. 最新训练日志尾部
  4. 最新训练指标 (results.csv)
  5. 磁盘空间

配置:
  服务器连接信息在仓库根目录 .server.conf.local
  （模板: Root/00_config/server.conf.example），运行脚本时密码交互输入。

用法:
  python scripts/check_server_status.py                      # 读取默认配置
  python scripts/check_server_status.py --config x.conf      # 指定配置文件
  python scripts/check_server_status.py --host 2.3.4.5       # 临时覆盖 host

依赖:
  pip install paramiko
"""

import argparse
import configparser
import getpass
import sys
import warnings
from pathlib import Path

# 屏蔽第三方库的弃用警告，保持输出干净
warnings.filterwarnings("ignore")

try:
    import paramiko
except ModuleNotFoundError:
    sys.exit(
        "错误: 缺少 paramiko 库，请先安装:\n\n"
        "  pip install paramiko\n"
    )


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / ".server.conf.local"
CONFIG_EXAMPLE = PROJECT_ROOT / "Root" / "00_config" / "server.conf.example"


def load_config(config_path: str) -> dict:
    """加载服务器配置，返回 {host, user, port, task_root, password}

    使用 RawConfigParser 避免密码中的 % 等字符被当作插值语法。
    """
    cp = configparser.RawConfigParser()
    path = Path(config_path)
    if not path.exists():
        sys.exit(
            f"[错误] 配置文件不存在: {path}\n"
            f"请从模板复制后填写服务器信息:\n"
            f"  cp {CONFIG_EXAMPLE} {path}\n"
            f"  # 然后编辑 {path} 填写 host / password 等"
        )
    cp.read(path, encoding="utf-8")
    if not cp.has_section("server"):
        sys.exit(f"[错误] 配置文件缺少 [server] 段: {path}")
    return {
        "host": cp.get("server", "host", fallback="").strip(),
        "user": cp.get("server", "user", fallback="root").strip(),
        "port": cp.getint("server", "port", fallback=22),
        "task_root": cp.get("server", "task_root", fallback="").strip(),
        "password": cp.get("server", "password", fallback="").strip(),
    }


# ---------------------------------------------------------------------------
# 远程查询
# ---------------------------------------------------------------------------

# 5 段查询，一段一条命令组
CHECKS = [
    ("1. 训练进程", """
echo "---- 进程列表 ----"
ps aux | grep "[y]olo detect train" || echo "无训练进程运行"
PID=$(pgrep -f "[y]olo detect train" | head -1)
if [ -n "$PID" ]; then
    echo ""
    echo "运行时长: $(ps -o etime= -p $PID)"
fi
"""),
    ("2. GPU 状态", """
nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
"""),
    ("3. 最新训练日志", """
LATEST=$(ls -t ${TASK_ROOT}/*/logs/train_*.log 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "日志文件: $LATEST"
    echo ""
    tail -n 15 "$LATEST"
else
    echo "未找到训练日志 (${TASK_ROOT}/*/logs/train_*.log)"
fi
"""),
    ("4. 训练指标", """
# 兼容两种目录布局: task/dataset/runs 和 task/数据集名/runs
LATEST=$(ls -t ${TASK_ROOT}/*/*/runs/detect/train*/results.csv 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "结果文件: $LATEST"
    echo ""
    tail -n 1 "$LATEST"
else
    echo "未找到 results.csv（训练可能刚开始或未完成第 1 个 epoch）"
fi
"""),
    ("5. 磁盘空间", """
df -h / /data 2>/dev/null | grep -v "^Filesystem"
"""),
]


def main():
    parser = argparse.ArgumentParser(description="查看训练服务器状态")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"配置文件路径 (默认: {DEFAULT_CONFIG})")
    parser.add_argument("--host", help="服务器 IP（覆盖配置文件）")
    parser.add_argument("--user", help="SSH 用户名（覆盖配置文件）")
    parser.add_argument("--port", type=int, help="SSH 端口（覆盖配置文件）")
    parser.add_argument("--task-root", help="训练任务目录（覆盖配置文件）")
    parser.add_argument("--password", help="服务器密码（不推荐，直接输入更安全）")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI 参数覆盖配置文件
    host = args.host or cfg["host"]
    user = args.user or cfg["user"]
    port = args.port or cfg["port"]
    task_root = args.task_root or cfg["task_root"]

    if not host:
        sys.exit(
            "[错误] 未配置服务器 IP\n"
            f"请编辑 {args.config} 填写: host = 1.2.3.4\n"
            "或运行时指定: --host 1.2.3.4"
        )

    password = args.password or cfg["password"]
    if not password:
        password = getpass.getpass(f"服务器密码 ({user}@{host}): ")

    print("=" * 60)
    print(f"  训练服务器状态")
    print(f"  {user}@{host}:{port}  任务目录: {task_root}")
    print("=" * 60)

    # 连接
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username=user, password=password, timeout=15)
    except paramiko.AuthenticationException:
        sys.exit("[错误] 认证失败，密码错误")
    except Exception as e:
        sys.exit(f"[错误] 连接失败: {e}")

    print(f"  已连接: {host}")

    try:
        for title, script in CHECKS:
            print(f"\n{'=' * 10} {title} {'=' * 10}")
            # task_root 注入远程环境
            full_cmd = f'TASK_ROOT="{task_root}"\n{script}'
            stdin, stdout, stderr = client.exec_command(full_cmd, timeout=60)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            if out:
                print(out.rstrip())
            if err:
                print(f"  [stderr] {err.rstrip()}")
    finally:
        client.close()

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
