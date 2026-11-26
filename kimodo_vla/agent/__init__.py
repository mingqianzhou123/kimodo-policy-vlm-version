"""A VLM (or a script) drives the robot one body command at a time; a planner turns each command into
the 40-d reference the tracker executes.

  commands   the written command set, its JSON schema and the validation of one command
  planner    the planner interface and the planner built from the existing Kimodo building blocks
  posture    pelvis height and torso pitch with the feet where they stand (new)
  loop       the step loop served to the simulator: observation -> commander -> planner -> tracker
  commander  where commands come from: a scripted sequence or a VLM
  episode_log  the per-episode record (layout, raw VLM output, targets, reference and measured end states)
  judge      which of the five links failed first, from an episode's logs alone; reach margins

Nothing here changes the behaviour of any existing policy.
"""
