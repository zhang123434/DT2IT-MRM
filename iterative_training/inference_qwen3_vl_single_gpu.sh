#!/bin/bash
#全面两条指令保证结果的可复现性
export NPU_COMPILE_TYPE=static  # 或 dynamic，取决于平台建议
export ASCEND_GLOBAL_LOG_LEVEL=3
# 日志目录
LOG_DIR="./logs"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 启动 8 个任务（每两个任务共用一张 NPU）
for id in {0..7}
do
    # npu_id=$((id / 2))  # 每两个任务共享一张 NPU
    ASCEND_RT_VISIBLE_DEVICES=$id python qwen3_vl_single_gpu.py $id > "$LOG_DIR/log_$id.txt" 2>&1 &
done

# 等待所有子进程结束
wait
