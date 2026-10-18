#!/usr/bin/env bash
# Build a TensorRT FP16 engine from the exported ONNX ON THE JETSON (engines are not portable between
# GPUs / TensorRT versions — never copy an engine built elsewhere) and measure latency for T-11 / PC-07.
#
#   ./build_trt_engine.sh starter_v1.onnx            # -> starter_v1_fp16.engine + starter_v1_fp16_bench.txt
#   MAX_BATCH=3 ./build_trt_engine.sh forklift_v1.onnx
#
# Before measuring: sudo nvpmodel -m 0 (MAXN SUPER on Orin Nano Super, JetPack 6.2) && sudo jetson_clocks
set -euo pipefail

ONNX=${1:?usage: $0 model.onnx}
MAX_BATCH=${MAX_BATCH:-1}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
ENGINE=${ONNX%.onnx}_fp16.engine
LOG=${ONNX%.onnx}_fp16_bench.txt

"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 \
  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:${MAX_BATCH}x3x640x640 \
  --memPoolSize=workspace:1024M --skipInference

# Steady-state latency at batch 1 (the robot runs one camera, 30 fps). PC-07: mean GPU compute <= 30 ms.
"$TRTEXEC" --loadEngine="$ENGINE" --shapes=images:1x3x640x640 --warmUp=2000 --duration=15 --useSpinWait \
  | tee "$LOG"

echo
echo "engine: $ENGINE ($(du -h "$ENGINE" | cut -f1))"
grep -E "Throughput|GPU Compute Time|Latency:" "$LOG" || true
echo "PC-07 passes when 'GPU Compute Time: ... mean' <= 30 ms AND the end-to-end node rate (pre + infer + post,"
echo "measured in the ROS 2 node) is >= 15 fps. Record both in docs/16-vision.md (T-11)."
