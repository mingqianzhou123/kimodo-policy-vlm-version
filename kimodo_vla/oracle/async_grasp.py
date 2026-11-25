"""Box grasp whose target may change while the arms move (experiment four).

``BoxGraspPolicy`` latches the box once per round, waits the delay and executes the whole
reach on that pose. This subclass keeps its walking, gating and lifting and changes only
where the reach aims and when that aim is refreshed:

- ``blocking``: the parent's behaviour; only the per-round delay may be drawn from
  measured latencies instead of being fixed.
- ``fixed``: a model asked every ``query_interval_s`` without waiting for earlier answers;
  each answer arrives one latency later and carries the box pose from when it was asked.
  Every replan aims at the newest answer that has arrived.
- ``anchored``: the model answers once (which object, where to hold it); from then on every
  replan aims at where the object is now.

The chunk already sent is always executed to its end; a new aim only shapes the next plan,
which starts from the reference SONIC has consumed. Once the hands have closed the aim is
frozen, as in the parent. The pose source is the simulator's truth unless a tracker is
given (``perception``), whose estimate then replaces the current pose for ``anchored``.
"""
import json

import numpy as np

from .grasp_policy import BoxGraspPolicy, WristPath, graspable, waypoints

UPDATES = ('blocking', 'fixed', 'anchored')


def latency_draws(samples, seed, count=400):
    """Seconds per call, drawn with replacement from measured latencies; the same for every update mode."""
    rng = np.random.default_rng(int(seed) + 9157)
    return rng.choice(np.asarray(samples, dtype=float), size=count)


class BoxLog:
    """Measured box poses by simulation time; ``at(t)`` is the last pose measured at or before t."""
    def __init__(self):
        self.times, self.com, self.quat = [], [], []

    def add(self, sample):
        if self.times and sample['time'] <= self.times[-1] + 1e-9:
            return
        self.times.append(float(sample['time']))
        self.com.append(list(sample['box_com']))
        self.quat.append(list(sample['box_quat']))

    def at(self, time):
        i = int(np.searchsorted(self.times, float(time) + 1e-6)) - 1
        if i < 0:
            raise ValueError(f'No box pose measured by {time}')
        return self.times[i], self.com[i], self.quat[i]


class AsyncBoxGraspPolicy(BoxGraspPolicy):
    def __init__(self, *args, target_update='anchored', latency_samples=None, query_interval_s=.5,
                 retarget_tolerance_m=.001, perception=None, **kwargs):
        if target_update not in UPDATES:
            raise ValueError(f'Unknown target update {target_update}')
        if query_interval_s <= 0 or retarget_tolerance_m < 0:
            raise ValueError('Positive query interval and non-negative retarget tolerance required')
        self.target_update = target_update
        self.latency_samples = None if latency_samples is None else list(map(float, latency_samples))
        self.query_interval = float(query_interval_s)
        self.retarget_tolerance = float(retarget_tolerance_m)
        self.perception = perception
        super().__init__(*args, **kwargs)
        if not self.stand_still:
            raise ValueError('Experiment four reaches from where the robot stands (--stand-still)')
        self.fixed_delay = self.latch_delay

    def reset(self, seed=None):
        response = super().reset(seed)
        self.boxes, self.episode_start, self.updates = BoxLog(), None, 0
        self.latencies = None if self.latency_samples is None else latency_draws(self.latency_samples, self.seed)
        self.anchor_time = self.anchor_root = None
        if self.perception is not None:
            self.perception.reset(self.seed)
        return response

    # ---- what the robot knows about the box ------------------------------------------------
    def latency(self, k):
        """Seconds the k-th call takes: the fixed delay, or the k-th draw from measured latencies."""
        return self.fixed_delay / 50 if self.latencies is None else float(self.latencies[k % len(self.latencies)])

    def first_answer(self):
        """When the robot first knows anything. With drawn latencies a later call of the
        pipelined ``fixed`` mode can come back before the first one."""
        first = self.latency(0)
        if self.target_update == 'fixed':
            k = 1
            while k * self.query_interval < first:
                first = min(first, k * self.query_interval + self.latency(k))
                k += 1
        return self.episode_start + first

    def belief(self, now):
        """(box centre, box quaternion, time the information is from), or None before any answer."""
        if self.target_update == 'fixed':
            k, best = 0, None
            while self.episode_start + k * self.query_interval <= now + 1e-9:
                asked = self.episode_start + k * self.query_interval
                if asked + self.latency(k) <= now + 1e-9:
                    best = asked
                k += 1
            if best is None:
                return None
            t, com, quat = self.boxes.at(best)
            return com, quat, t
        if now < self.first_answer() - 1e-9:
            return None
        if self.perception is not None:
            return self.perception.estimate(now)
        t, com, quat = self.boxes.at(now)
        return com, quat, t

    # ---- rounds ----------------------------------------------------------------------------
    def begin_episode(self, first):
        self.episode_start = first['time']
        self.boxes.add(first)
        super().begin_episode(first)

    def begin_round(self, sample):
        """Round 0 starts as the parent's (the first picture is taken now). Later rounds of the
        asynchronous modes start at once from what is known; blocking rounds wait a new delay."""
        index = 0 if self.round is None else self.round.index + 1
        if self.target_update == 'blocking':
            self.latch_delay = int(round(self.latency(index) * 50))
            return super().begin_round(sample)
        if index == 0:
            self.latch_delay = int(np.ceil((self.first_answer() - self.episode_start) * 50 - 1e-6))
            return super().begin_round(sample)
        self.latch_delay = 0
        known = self.belief(sample['time'])
        super().begin_round(sample if known is None else {**sample, 'box_com': known[0], 'box_quat': known[1]})

    def advance(self, samples):
        for sample in samples:
            self.boxes.add(sample)
        if self.perception is not None:
            self.perception.observe(samples)
        return super().advance(samples)

    def anchor(self, obs):
        r = self.round
        if self.target_update != 'blocking':
            com, quat, info_time = self.belief(obs['time'])
            r.box_pos, r.box_quat = list(com), list(quat)
            r.graspable = graspable(com, quat, self.cfg)
            r.notes['target_info_time'] = info_time
        else:
            r.notes['target_info_time'] = r.latch_time
        self.anchor_time, self.anchor_root = obs['time'], list(obs['root_pos'][:2])
        super().anchor(obs)
        self.log('update', update=self.updates, target_box_pos=r.box_pos, target_box_quat=r.box_quat,
                 info_time=r.notes['target_info_time'], now=obs['time'], first=True)

    def correct(self, obs):
        if self.target_update != 'blocking' and self.round.phase in ('raise', 'advance'):
            self.retarget(obs)
        super().correct(obs)

    def retarget(self, obs):
        """Aim this replan at the current belief: same timing, new waypoints, new gate target."""
        r, c, now = self.round, self.cfg, obs['time']
        known = self.belief(now)
        if known is None:
            return
        com, quat, info_time = known
        w = waypoints(com, quat, self.anchor_root, c)
        path = WristPath(self.anchor_time, *self.rest)
        path.to(w['via'], w['via_rotation'], c['raise_s'])
        path.to(w['pre'], w['rotation'], c['advance_s'])
        moved = float(np.abs(w['pre'] - self.way['pre']).max())
        self.way, self.path = w, path
        r.box_pos, r.box_quat = list(com), list(quat)
        r.notes['target_info_time'] = info_time
        changed = moved > self.retarget_tolerance
        if changed:
            self.updates += 1
            r.event = {**r.event, 'id': f'round-{r.index}-u{self.updates}', 'positions': w['pre'].tolist(),
                       'rotations': w['rotation'].tolist()}
            r.notes.update(opening_m=w['opening_m'], face_yaw_rad=w['face_yaw_rad'],
                           wrist_targets={k: w[k].tolist() for k in ('via', 'pre', 'squeeze', 'lift')})
        self.log('update', update=self.updates, changed=changed, moved_m=moved, target_box_pos=r.box_pos,
                 target_box_quat=r.box_quat, info_time=info_time, now=now)

    def grasp_info(self):
        info = super().grasp_info()
        info['target_update'] = self.target_update
        return info


def load_latencies(path):
    """Latencies in seconds from a JSON file: a list, or an object with a 'latency_s' list."""
    data = json.loads(open(path).read())
    values = data['latency_s'] if isinstance(data, dict) else data
    if not values or min(values) <= 0:
        raise ValueError('Latency samples must be positive')
    return values
