"""Diagnostic: map MUJOCO_EGL_DEVICE_ID -> physical GPU (the one whose memory grows)."""
import os, subprocess, sys

idx = sys.argv[1]
os.environ["MUJOCO_EGL_DEVICE_ID"] = idx
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"
ok, err = True, ""
try:
    import mujoco
    ctx = mujoco.GLContext(1024, 1024)
    ctx.make_current()
except Exception as e:
    ok, err = False, f"{type(e).__name__}:{str(e)[:60]}"
mem = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]
).decode().strip().replace("\n", " | ")
print(f"MUJOCO_EGL_DEVICE_ID={idx} ok={ok} {err}  ->  {mem}")
