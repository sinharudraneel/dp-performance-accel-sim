
export CUDA_VERSION="11.7"; export CUDA_VISIBLE_DEVICES="0" ; export TRACES_FOLDER=/root/dp-performance-accel-sim/hw_run/traces/device-0/11.7/simpletorch2/NO_ARGS/traces; CUDA_INJECTION64_PATH=/root/dp-performance-accel-sim/util/tracer_nvbit/tracer_tool/tracer_tool.so /root/dp-performance-accel-sim/gpu-app-collection/src/..//bin/11.7/release/simpletorch2  ; /root/dp-performance-accel-sim/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing /root/dp-performance-accel-sim/hw_run/traces/device-0/11.7/simpletorch2/NO_ARGS/traces/kernelslist ; rm -f /root/dp-performance-accel-sim/hw_run/traces/device-0/11.7/simpletorch2/NO_ARGS/traces/*.trace ; rm -f /root/dp-performance-accel-sim/hw_run/traces/device-0/11.7/simpletorch2/NO_ARGS/traces/kernelslist 