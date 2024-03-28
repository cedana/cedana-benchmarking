FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt install -y ocaml libelf-dev

COPY cdpAdvancedQuicksort cdpAdvancedQuicksort
COPY libcedana-gpu.so /usr/lib64/libcedana-gpu.so


CMD ["bash", "-c", "LD_PRELOAD=/usr/lib64/libcedana-gpu.so ./cdpAdvancedQuicksort"]
