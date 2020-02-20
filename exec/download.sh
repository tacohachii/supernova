#!/bin/bash

# bash download.sh

# 配列の作成
mass_name_arr=()
for ((i=54; i <= 300; i++)); do
  mass_name_arr+=(`echo "scale=1; $i*2.0/10.0" | bc`)
done
mass_name_arr+=("75.0")

# ダウンロード
for name in ${mass_name_arr[@]}; do
  curl -fs --create-dirs -o "./data/woosley/s$name.gz" -OL https://2sn.org/stellarevolution/solar/s${name}.gz
  curl -fs --create-dirs -o "./data/woosley/u$name.gz" -OL https://2sn.org/stellarevolution/ultra/u${name}.gz
done

# 解凍
find ./data/woosley -type f -name "*.gz" -exec gzip -fd {} \;

echo "FINISH"
exit 0

