#!/bin/bash

set -euo pipefail
shopt -s nullglob

IN_DIR="videos"
OUT_ROOT="video_images"

for inpath in "$IN_DIR"/*.mp4; do
  base="$(basename "$inpath" .mp4)"
  outdir="$OUT_ROOT/$base"
  echo ">> Extracting frames from $inpath -> $outdir"

  ffmpeg -i "$inpath" \
    -q:v 2 \
    -start_number 0 \
    "$outdir/%05d.jpg"
done

echo "All done."