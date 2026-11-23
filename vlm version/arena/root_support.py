"""Process-local static support counterfactual; never changes a reference or robot."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def install_root_support():
    from isaaclab.scene import InteractiveScene
    import isaaclab.sim as sim_utils
    from pxr import Usd,UsdGeom,UsdPhysics
    import omni.usd
    original=InteractiveScene.__init__
    mode=os.environ['SONIC_ROOT_SUPPORT'];run=Path(os.environ['ORACLE_RUN'])
    spec=json.loads((ROOT/'configs/root-dynamics.json').read_text())

    def init(scene,*args,**kwargs):
        original(scene,*args,**kwargs)
        stage=omni.usd.get_context().get_stage()
        robot=stage.GetPrimAtPath('/World/envs/env_0/Robot')
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
        feet=[]
        for body in Usd.PrimRange(robot):
            if body.GetName() not in ('left_ankle_roll_link','right_ankle_roll_link'):continue
            for prim in Usd.PrimRange(body):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    b=cache.ComputeRelativeBound(prim,body).ComputeAlignedRange()
                    feet.append(dict(body=body.GetName(),path=str(prim.GetPath()),type=prim.GetTypeName(),minimum=list(b.GetMin()),maximum=list(b.GetMax())))
        if not feet:raise ValueError('No foot collision geometry')
        for side in ('left','right'):
            low=min(x['minimum'][2] for x in feet if x['body'].startswith(side))
            if abs(low-spec['sole_min_z_m'])>1e-5:raise ValueError(('Unexpected sole geometry',side,low))
        record=dict(mode=mode,spec=spec,feet=feet,platform=None)
        if mode=='raised':
            h=spec['support_height_m'];size=spec['platform_size_m'];center=[*spec['platform_xy_m'],h-size[2]/2]
            cfg=sim_utils.CuboidCfg(size=tuple(size),collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True,contact_offset=0.02,rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.,friction_combine_mode='max',restitution_combine_mode='max'))
            path='/World/RootSupportDiagnostic';cfg.func(path,cfg,translation=tuple(center))
            prim=stage.GetPrimAtPath(path);cache.Clear();b=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            record['platform']=dict(path=path,center=center,size=size,minimum=list(b.GetMin()),maximum=list(b.GetMax()),config=cfg.to_dict())
            if abs(b.GetMax()[2]-h)>1e-6:raise ValueError('Support top differs')
        # Local scene changes only; shared assets are never saved.
        from arena.actuator_audit import plain
        (run/'root-scene-audit.json').write_text(json.dumps(plain(record),indent=2))
    InteractiveScene.__init__=init
