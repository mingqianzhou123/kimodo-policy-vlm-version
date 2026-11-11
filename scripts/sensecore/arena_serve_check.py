"""Load a run's last checkpoint the way serve_arena.py does and answer one Arena request."""
import base64, glob, sys
import numpy as np
from kimodo_vla.benchmarks.arena.policy import make_kimodo_policy

checkpoint = sorted(glob.glob(sys.argv[1] + "/checkpoint_*"), key=lambda p: int(p.rsplit("_", 1)[1]))[-1]
policy = make_kimodo_policy(checkpoint, sys.argv[2], controller="sonic")
policy.reset(2)
image = np.zeros((480, 640, 3), np.uint8)
state = np.zeros((100, 64), np.float32)
state[:, 0] = state[:, 4] = 1.0
request = {"task": "Put the hammer from the right table into the basket on the left table.",
           "observation": {"images": {"front": {"shape": [480, 640, 3], "dtype": "uint8", "data_b64": base64.b64encode(image.tobytes()).decode()}},
                           "state_history": state.tolist()}}
action = np.asarray(policy(request)["action_chunk"])
print("SERVE_OK" if action.ndim == 2 and action.shape[1] == 40 and np.isfinite(action).all() else "SERVE_BAD", checkpoint, action.shape)
