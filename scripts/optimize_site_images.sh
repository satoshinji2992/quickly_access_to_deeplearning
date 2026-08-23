#!/bin/bash
# 站点展示图优化：PNG → WebP（长边 ≤1600，q82），PNG 保留作为回退。
# 产物与源文件同名，仅扩展名不同。重复运行幂等。
set -euo pipefail
cd "$(dirname "$0")/.."

MAXEDGE=1600
Q=82

optimize() {
  local png="$1" webp="$2"
  local w h
  w=$(sips -g pixelWidth "$png" | awk '/pixelWidth/{print $2}')
  h=$(sips -g pixelHeight "$png" | awk '/pixelHeight/{print $2}')
  local args=()
  if [ "$w" -ge "$h" ] && [ "$w" -gt "$MAXEDGE" ]; then
    args+=(-resize "$MAXEDGE" 0)
  elif [ "$h" -gt "$MAXEDGE" ]; then
    args+=(-resize 0 "$MAXEDGE")
  fi
  cwebp -quiet -m 6 -q "$Q" ${args[@]+"${args[@]}"} "$png" -o "$webp"
  echo "$(basename "$webp")  $(du -k "$webp" | cut -f1) KB  (PNG: $(du -k "$png" | cut -f1) KB)"
}

for f in site/static/assets/figures/*.png site/static/assets/og.png; do
  optimize "$f" "${f%.png}.webp"
done
