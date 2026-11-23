"""Strict image-point protocol and calibrated optical-Z backprojection."""
import json
import numpy as np

MOTION_TEXT = 'Reach both hands toward the targets and hold still.'


def parse_points(text):
    text = text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    result = json.loads(text)
    if set(result) != {'left', 'right', 'motion_text'}:
        raise ValueError('Expected left, right and motion_text only')
    if result['motion_text'] != MOTION_TEXT:
        raise ValueError('Unsupported motion phrase')
    for side in ('left', 'right'):
        point = result[side]
        if point is None:
            raise ValueError(f'{side} target is not visible')
        if not isinstance(point,list) or len(point)!=2 or any(type(x) not in (int,float) for x in point):
            raise ValueError('Point must be two numeric coordinates')
        if not np.isfinite(point).all() or np.any(np.array(point)<0) or np.any(np.array(point)>1000):
            raise ValueError('Point outside normalized image range [0,1000]')
    return result


def backproject(point, depth, K, camera_position, camera_rotation):
    """Qwen [0,1000] spans pixel centers 0..W-1/H-1; no crop or letterbox."""
    depth=np.asarray(depth);K=np.asarray(K);R=np.asarray(camera_rotation);p=np.asarray(camera_position)
    if depth.ndim!=2 or K.shape!=(3,3) or R.shape!=(3,3) or p.shape!=(3,):
        raise ValueError('Invalid camera shapes')
    if not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('Invalid intrinsic calibration')
    if not np.isfinite(p).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1):
        raise ValueError('Invalid optical-to-world rotation')
    h,w=depth.shape
    uv=np.asarray(point,dtype=float)/1000*np.array([w-1,h-1])
    if not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>[w-1,h-1]):
        raise ValueError('Point outside image')
    x,y=np.rint(uv).astype(int)
    z=float(depth[y,x])
    if not np.isfinite(z) or not .1<=z<=3.:
        raise ValueError('Missing or out-of-range depth at selected pixel')
    patch=depth[max(y-1,0):min(y+2,h),max(x-1,0):min(x+2,w)]
    if not np.isfinite(patch).all() or np.ptp(patch)>.05:
        raise ValueError('Selected pixel is on a depth discontinuity')
    # Use the sampled pixel ray rather than a different subpixel ray.
    optical=z*np.linalg.solve(K,np.array([x,y,1.]))
    return R@optical+p, {'pixel':[int(x),int(y)],'normalized_point':list(point),'depth_z_m':z}


def load_capture(path):
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k].copy() for k in f.files}
    if str(data['depth_kind'])!='distance_to_image_plane' or str(data['camera_convention'])!='ros_optical':
        raise ValueError('Expected metric optical Z and ROS optical coordinates')
    if data['rgb'].shape[:2]!=data['depth'].shape or data['rgb'].dtype!=np.uint8:
        raise ValueError('RGB and depth are not aligned')
    return data
