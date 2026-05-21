#!/bin/bash
# Shared body for box_transport sbatch runs. Sourced by the per-algo,
# per-mode wrappers.
#
# Required env vars:
#   ALGO  ∈ {ppo, sapg, epo}
#   MODE  ∈ {scratch, warmstart, curriculum}
#
# MODE drives the difference between the run families:
#   scratch     — random init, no checkpoint.        <algo>_NN_scratch
#   warmstart   — load velocity_tracking model_1500.  <algo>_NN_warmstart
#   curriculum  — warmstart + the 3-stage reverse curriculum enabled,
#                 longer budget (8000 iters).         <algo>_NN_curriculum
# ``+run_tag=$MODE`` is appended so the run dir / wandb name carry it.

set -euo pipefail

: "${ALGO:?must be set: ppo|sapg|epo}"
: "${MODE:?must be set: scratch|warmstart|curriculum}"

IMAGE=/home/danielc174/IsaacSim/isaacsim.sif
ISAACLAB=/home/danielc174/IsaacLab
PROJECT=/home/danielc174/projects/humanoid-rl
CACHE=/home/danielc174/isaac_cache
SCRATCH=/scratch2/danielc174

mkdir -p "$CACHE"/{kit,ov,pip,glcache,computecache,logs,data,documents}

# Mode-specific Hydra args.
EXTRA_ARGS=( "+run_tag=$MODE" )
if [ "$MODE" = "warmstart" ] || [ "$MODE" = "curriculum" ]; then
    CKPT="/scratch2/danielc174/humanoid-rl/outputs/velocity_tracking/${ALGO}/${ALGO}_00/checkpoints/model_1500.pt"
    [ -f "$CKPT" ] || { echo "[sbatch] CKPT NOT FOUND: $CKPT"; exit 1; }
    EXTRA_ARGS+=( "checkpoint=$CKPT" )
    echo "[sbatch] warm-start ckpt: $CKPT"
fi
if [ "$MODE" = "curriculum" ]; then
    # Enable the 3-stage reverse curriculum and give it a longer budget so
    # there is room to progress through all stages and still train on the
    # full task.
    EXTRA_ARGS+=( "++task.curriculum_enabled=true" "++task.max_iterations=8000" )
    echo "[sbatch] curriculum enabled, max_iterations=8000"
fi

echo "[sbatch] node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} algo=$ALGO mode=$MODE"

singularity exec --cleanenv --writable-tmpfs --nv \
  --bind "$ISAACLAB":/IsaacLab \
  --bind "$PROJECT":"$PROJECT" \
  --bind "$SCRATCH":"$SCRATCH" \
  --bind "$CACHE"/kit:/isaac-sim/kit/cache \
  --bind "$CACHE"/ov:/root/.cache/ov \
  --bind "$CACHE"/pip:/root/.cache/pip \
  --bind "$CACHE"/glcache:/root/.cache/nvidia/GLCache \
  --bind "$CACHE"/computecache:/root/.nv/ComputeCache \
  --bind "$CACHE"/logs:/root/.nvidia-omniverse/logs \
  --bind "$CACHE"/data:/root/.local/share/ov/data \
  --bind "$CACHE"/documents:/root/Documents \
  --pwd "$PROJECT" \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --env HEADLESS=1 \
  --env GIT_PYTHON_REFRESH=quiet \
  "$IMAGE" \
  /IsaacLab/isaaclab.sh -p scripts/train.py \
    task=box_transport \
    "algo=$ALGO" \
    wandb.mode=online \
    "${EXTRA_ARGS[@]}"
