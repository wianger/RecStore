#!/bin/bash

LOCK_FILE="/tmp/xmh_gsw_lock"

while [ -e "$LOCK_FILE" ]; 
do
    echo "锁文件已存在，脚本正在执行中，退出"
    sleep 1
done

# 创建锁文件
touch "$LOCK_FILE"

# 执行操作
echo "脚本开始执行..."

bash run.sh ./benchmark-optimized 1 1 1

echo "脚本执行完成."

# 删除锁文件
rm -f "$LOCK_FILE"
echo "锁文件已删除，脚本执行完毕."