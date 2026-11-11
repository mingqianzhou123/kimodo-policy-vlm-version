#!/usr/bin/env bash
# EGL for MuJoCo in ACP jobs (re-runnable, on the dev instance). The ACP images have no libEGL, so
# `import mujoco` with MUJOCO_GL=egl fails. This builds $KVLA_HOME/envs/egl_loader: the glvnd loader
# (libEGL, libOpenGL, libGLdispatch, nothing else, so training's libraries are untouched) and vendor files
# for NVIDIA's libEGL_nvidia.so.0, which the driver mounts into a job when the job sets
# NVIDIA_DRIVER_CAPABILITIES=all, and for Mesa (CPU rendering; works but ~400x slower than BiGym's physics).
# bigym_run.sh uses the loader with the NVIDIA vendor only, and render_probe.py checks it got a GPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
H=$KVLA_HOME
CF=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
G=$H/envs/gl
[ -e $G/lib/libOpenGL.so.0 ] || /root/anaconda3/bin/conda create -y -q -p $G --override-channels -c $CF mesalib libegl libglvnd libgl libopengl
E=$H/envs/egl_loader
mkdir -p $E/lib $E/vendor_nvidia $E/vendor_mesa
for f in libEGL.so.1 libGLdispatch.so.0 libOpenGL.so.0; do cp -L $G/lib/$f $E/lib/; done
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > $E/vendor_nvidia/10_nvidia.json
sed "s#\"libEGL_mesa.so.0\"#\"$G/lib/libEGL_mesa.so.0\"#" $G/share/glvnd/egl_vendor.d/50_mesa.json > $E/vendor_mesa/50_mesa.json
echo EGL_LOADER_READY
