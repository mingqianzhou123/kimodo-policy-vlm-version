"""Explicit world Z-up <-> Kimodo Y-up coordinates and independent XML FK."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

BASIS = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
SIDES = ('left', 'right')


def rotation_wxyz(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


class Frame:
    def __init__(self, origin, heading):
        self.origin = np.asarray(origin, dtype=float).copy()
        self.origin[2] = 0.
        self.heading = np.asarray(heading, dtype=float)
        self.world_to_model = BASIS @ self.heading.T

    def points(self, world):
        return (np.asarray(world) - self.origin) @ self.world_to_model.T

    def world(self, model):
        return np.asarray(model) @ self.world_to_model + self.origin

    def rotations(self, world):
        return self.world_to_model @ world @ BASIS.T


class XmlFK:
    """Independent FK for this fixed G1 MJCF using XML offsets and hinge axes.

    All hinges in this asset have zero pivot offsets. Reject other schemas;
    no MuJoCo package is needed in the Isaac or policy process.
    """
    def __init__(self, xml, joint_names):
        tree = ET.parse(xml).getroot()
        self.joint_names = tuple(joint_names)
        defaults = {}
        def read_defaults(node, inherited):
            values = dict(inherited)
            joint = node.find('joint')
            if joint is not None:
                values.update(joint.attrib)
            defaults[node.get('class', '')] = values
            for child in node.findall('default'):
                read_defaults(child, values)
        for node in tree.findall('default'):
            read_defaults(node, {})
        self.nodes = []
        self.joint_limits = np.full((len(self.joint_names),2),[-np.inf,np.inf])
        def walk(body, parent):
            idx = len(self.nodes)
            pos = np.fromstring(body.get('pos','0 0 0'),sep=' ')
            rot = rotation_wxyz(np.fromstring(body.get('quat','1 0 0 0'),sep=' '))
            joints = []
            for joint in body.findall('joint'):
                attrs = {**defaults.get(joint.get('class',''),{}), **joint.attrib}
                if attrs.get('type','hinge') != 'hinge' or attrs.get('pos','0 0 0') != '0 0 0':
                    raise ValueError('Unsupported joint schema in fixed G1 XML')
                axis = np.fromstring(attrs.get('axis','0 0 1'),sep=' ')
                axis /= np.linalg.norm(axis)
                joints.append((self.joint_names.index(attrs['name']),axis))
                if 'range' in attrs:
                    self.joint_limits[self.joint_names.index(attrs['name'])] = np.fromstring(attrs['range'],sep=' ')
            self.nodes.append((body.get('name'),parent,pos,rot,joints))
            for child in body.findall('body'):
                walk(child,idx)
        pelvis = tree.find("worldbody/body[@name='pelvis']")
        if pelvis is None or pelvis.find('freejoint') is None:
            raise ValueError('Expected floating pelvis')
        walk(pelvis,-1)
        names = [x[0] for x in self.nodes]
        self.body_names = names
        self.wrist_ids = [names.index(f'{side}_wrist_yaw_link') for side in SIDES]

    def __call__(self, joints, root, rotation, body_names=None):
        q = np.asarray(joints)
        positions, rotations = [], []
        for name,parent,offset,rest,hinges in self.nodes:
            if parent == -1:
                p,r = np.asarray(root),np.asarray(rotation)
            else:
                p = positions[parent]+rotations[parent] @ offset
                r = rotations[parent] @ rest
                for index,axis in hinges:
                    r = r @ Rotation.from_rotvec(axis*q[index]).as_matrix()
            positions.append(p); rotations.append(r)
        ids = self.wrist_ids if body_names is None else [self.body_names.index(n) for n in body_names]
        return np.array(positions)[ids],np.array(rotations)[ids]
