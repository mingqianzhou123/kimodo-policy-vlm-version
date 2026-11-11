#!/usr/bin/env bash
# Start command for every ACP job:  bash <KVLA_HOME>/acp_job.sh <name> <command...>
# Output goes to $KVLA_HOME/logs/acp/<name>_<UTC time>.log, so results survive the job.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NAME=$1; shift
mkdir -p $KVLA_HOME/logs/acp
LOG=$KVLA_HOME/logs/acp/${NAME}_$(date -u +%Y%m%d_%H%M%S).log
{
  echo "== $NAME on $(hostname) at $(date -u)"; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  echo "== $*"; "$@"; rc=$?; echo "== exit $rc"; exit $rc
} 2>&1 | tee $LOG
exit ${PIPESTATUS[0]}
