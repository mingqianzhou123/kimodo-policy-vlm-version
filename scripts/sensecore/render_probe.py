"""Print which GL renderer MuJoCo's EGL backend gets, and render one BiGym frame. Exit 0 only for a GPU renderer."""
import sys, time
import numpy as np
from bigym.loco import make

env = make("move_plate")
ts = env.reset(seed=0)
rgb = np.asarray(ts.rgb_obs)
from OpenGL import GL
renderer = GL.glGetString(GL.GL_RENDERER)
renderer = renderer.decode() if renderer else "unknown"
start = time.time()
action = np.zeros(env.action_space.shape, np.float32)
for _ in range(20):
    env.step(action)
rate = 20 / (time.time() - start)
print(f"GL_RENDERER {renderer} | rgb {rgb.shape} mean {rgb.mean():.1f} | {rate:.1f} env steps/s", flush=True)
sys.exit(0 if ("NVIDIA" in renderer or "A800" in renderer) and rgb.mean() > 1 else 3)
