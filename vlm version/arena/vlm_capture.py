"""One synchronized RGB-D capture; simulator truth stays in a separate evaluator file."""
from pathlib import Path
import numpy as np
from kimodo_vla.oracle.geometry import rotation_wxyz


def capture(env,run,camera_name='front_camera'):
    camera=env.scene[camera_name]
    camera.cfg.update_latest_camera_pose=True
    env.sim.forward()
    # Render at the paused current simulator time and refresh all camera data.
    # Newly spawned USD markers need renderer synchronization; physics remains
    # paused so RGB, depth, extrinsics and proprioception share the same time.
    for _ in range(6):
        env.sim.render()
    # update(0, force_recompute=True) only refreshes OUTDATED buffers in
    # IsaacLab 2.2; at a paused timestamp it otherwise returns the old image.
    camera.reset()
    camera.update(0.,force_recompute=True)
    data=camera.data
    cpu=lambda x:x[0].detach().cpu().numpy().copy()
    rgb=cpu(data.output['rgb'])[...,:3]
    depth=cpu(data.output['distance_to_image_plane']).squeeze(-1)
    path=Path(run)/'vlm-capture.npz'
    np.savez_compressed(path,rgb=rgb,depth=depth,K=cpu(data.intrinsic_matrices),
        camera_position=cpu(data.pos_w),camera_rotation=rotation_wxyz(cpu(data.quat_w_ros)),
        time=float(env.sim.current_time),camera_name=camera_name,
        depth_kind='distance_to_image_plane',camera_convention='ros_optical')
    return {'capture_path':str(path),'time':float(env.sim.current_time),'camera_name':camera_name}
