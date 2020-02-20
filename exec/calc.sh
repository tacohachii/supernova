#!/bin/bash

# コード実行時は以下のコードを打つ（ファイル名は下で指定する必要ある）
# bash calc.sh >& calc.log 2> calc_error.log

# データの初期化
rm -f ./output/main.dat
rm -f .calc_error.log
rm -f .calc.log

# ファイル名の取得
files="./data/mesa/*"
file_arr=()
for filepath in $files; do
  if [ -f $filepath ] ; then
    file_arr+=("${filepath##*/}")
  fi
done

# 処理実行
for filename in ${file_arr[@]}; do
  python3 main.py mesa $filename
  echo "Done $filename"
done

# plot処理
  python3 plot.py

echo "FINISH"
exit 0
