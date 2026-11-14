"""Check local physics and RGB rendering without external scene downloads."""
from pathlib import Path
import json
import warp as wp
print("PRELOADED_WARP", wp.__version__, wp.__file__, flush=True)
wp.init()
from isaaclab.app import AppLauncher

RUN = Path(__file__).resolve().parent
runtime_root = RUN.parents[1] / "data/cache/sim-runtime"
launcher = AppLauncher(
    headless=True, enable_cameras=True, device="cuda:0",
    kit_args=f"--allow-root --portable-root {runtime_root} "
             "--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0",
)
app = launcher.app

import numpy as np
import torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg

try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    floor = sim_utils.CuboidCfg(
        size=(10, 10, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    floor.func("/World/Floor", floor, translation=(0, 0, -0.05))
    light = sim_utils.DomeLightCfg(intensity=2000)
    light.func("/World/Light", light)
    cube = RigidObject(RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 1)),
    ))
    camera = Camera(CameraCfg(
        prim_path="/World/Camera", update_period=0, height=240, width=320,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
            focal_length=24, horizontal_aperture=20.955, clipping_range=(0.1, 1000),
        ),
    ))
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[3., 3., 3.]], device=sim.device),
        torch.tensor([[0., 0., 0.3]], device=sim.device),
    )
    initial_z = float(cube.data.root_pos_w[0, 2])
    for _ in range(180):
        sim.step()
        cube.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt())
    final_z = float(cube.data.root_pos_w[0, 2])
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
    assert 0.10 < final_z < 0.16 and initial_z > 0.9, (initial_z, final_z)
    assert rgb.shape == (240, 320, 3) and np.isfinite(rgb).all() and rgb.std() > 1
    Image.fromarray(rgb.astype(np.uint8)).save(RUN / "smoke-rgb-r4.png")
    result = {"passed": True, "steps": 180, "initial_cube_z": initial_z,
              "final_cube_z": final_z, "rgb_shape": list(rgb.shape),
              "rgb_std": float(rgb.std()), "meaning": "physics and RGB only, not task evaluation"}
    (RUN / "smoke-result-r4.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
finally:
    app.close(wait_for_replicator=False)
