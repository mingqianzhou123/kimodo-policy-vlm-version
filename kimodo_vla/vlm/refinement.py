"""Crop coordinate maps and marker-specific RGB verification; no scene truth."""
import json
import numpy as np

CROP_SIZES=(192,256)
MAX_DISAGREEMENT_PX=6.


def crop_box(point,image_size,size):
    w,h=image_size
    if w<2 or h<2:raise ValueError('Image too small')
    xy=np.asarray(point,dtype=float)/1000*np.array([w-1,h-1])
    cw,ch=min(size,w),min(size,h)
    x=int(np.clip(round(xy[0]-(cw-1)/2),0,w-cw))
    y=int(np.clip(round(xy[1]-(ch-1)/2),0,h-ch))
    return (x,y,x+cw,y+ch)


def parse_crop_point(text):
    text=text.strip()
    if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
    value=json.loads(text)
    if set(value)!={'point'}:raise ValueError('Expected crop point only')
    p=value['point']
    if p is None:raise ValueError('Target not visible in crop')
    if not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) for v in p):
        raise ValueError('Invalid crop point')
    if not np.isfinite(p).all() or np.any(np.array(p)<0) or np.any(np.array(p)>1000):
        raise ValueError('Crop point outside [0,1000]')
    return p


def crop_to_full(point,box):
    x,y,x2,y2=box
    return np.array([x,y],dtype=float)+np.asarray(point)/1000*np.array([x2-x-1,y2-y-1])


def verify_marker_pixel(rgb,point,side):
    """Only verify the model's point; do not find/snap to a colored centroid."""
    h,w=rgb.shape[:2];x,y=np.rint(point).astype(int)
    if not (1<=x<w-1 and 1<=y<h-1):raise ValueError('Target too close to image boundary')
    patch=rgb[y-1:y+2,x-1:x+2,:3].astype(float)
    channel={'left':0,'right':2}[side]
    other=np.max(patch[...,[i for i in range(3) if i!=channel]],axis=-1)
    valid=(patch[...,channel]>=80)&(patch[...,channel]>=1.5*(other+1))
    fraction=float(valid.mean())
    if fraction<5/9:raise ValueError(f'{side}: predicted pixel is not on the expected colored marker')
    return fraction


def combine_and_verify(rgb,points,side):
    points=np.asarray(points,dtype=float)
    if points.shape!=(2,2) or not np.isfinite(points).all():raise ValueError('Two finite crop points required')
    disagreement=float(np.linalg.norm(points[0]-points[1]))
    if disagreement>MAX_DISAGREEMENT_PX:raise ValueError(f'{side}: crop estimates disagree by {disagreement:.2f} px')
    checks=[verify_marker_pixel(rgb,p,side) for p in points]
    final=points.mean(0);checks.append(verify_marker_pixel(rgb,final,side))
    return final,{'disagreement_px':disagreement,'color_patch_fractions':checks}
