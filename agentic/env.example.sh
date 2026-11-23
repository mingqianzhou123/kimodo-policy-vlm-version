# Copy to agentic/env.local.sh (ignored by git), fill in for this machine, then:
#   source agentic/env.local.sh
export KVLA_SIM_CACHE=        # required: directory holding envs/, sources/, sim-assets/
export KVLA_CHECKPOINTS=      # optional: model weights; default <repo>/checkpoints
export KVLA_AGENTIC_RUNS=     # optional: experiment outputs; default agentic/runs. Use persistent storage.
