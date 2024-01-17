#!/bin/bash

# 设置本地源目录和远程目标目录
local_source="/home/xieminhui/RecStore/src"
remote_destination="3090:/home/xieminhui/RecStore1/src"

# 设置需要排除同步的目录
exclude_directory="/home/xieminhui/RecStore/src/framework_adapters/torch/kg/dgl-0.9.1"

# 循环执行 rsync 命令
while true
do
  # 使用 rsync 进行同步，通过 --exclude 参数排除指定目录
  rsync -av --exclude="$exclude_directory" "$local_source" "$remote_destination"
  print sync
  # 等待一段时间，可以根据需求调整
  sleep 60
done
