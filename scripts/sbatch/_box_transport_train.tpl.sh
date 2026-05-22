#!/bin/bash
# Shared body for box_transport sbatch runs. Sourced by the per-algo,
# per-mode wrappers.
#
# Required env vars:
#   ALGO  ∈ {ppo, sapg, epo}
#   MODE  ∈ {scratch, warmstart, phaseA, phaseB}
#
# MODE drives the run family:
#   scratch    — random init, no checkpoint.            <algo>_NN_scratch
#   warmstart  — load velocity_tracking model_1500.      <algo>_NN_warmstart
#   phaseA     — warmstart from velocity_tracking AND
#                assisted_grasp_start=true (robot at the
#                table, arms pre-posed around the box).  <algo>_NN_phaseA
#   phaseB     — the full task, warmstart from the most
#                recent phaseA run's final checkpoint.   <algo>_NN_phaseB
# ``+run_tag=$MODE`` is appended so the run dir / wandb name carry it.

set -euo pipefail

: "${ALGO:?must be set: ppo|sapg|epo}"
: "${MODE:?must be set: scratch|warmstart|phaseA|phaseB}"

IMAGE=/home/danielc174/IsaacSim/isaacsim.sif
ISAACLAB=/home/danielc174/IsaacLab
PROJECT=/home/danielc174/projects/humanoid-rl
CACHE=/home/danielc174/isaac_cache
SCRATCH=/scratch2/danielc174

mkdir -p "$CACHE"/{kit,ov,pip,glcache,computecache,logs,data,documents}

# Mode-specific Hydra args.
OUT=/scratch2/danielc174/humanoid-rl/outputs
EXTRA_ARGS=( "+run_tag=$MODE" )
case "$MODE" in
  warmstart|phaseA)
    CKPT="$OUT/velocity_tracking/${ALGO}/${ALGO}_00/checkpoints/model_1500.pt"
    [ -f "$CKPT" ] || { echo "[sbatch] velocity ckpt NOT FOUND: $CKPT"; exit 1; }
    EXTRA_ARGS+=( "checkpoint=$CKPT" )
    [ "$MODE" = "phaseA" ] && EXTRA_ARGS+=( "++task.assisted_grasp_start=true" )
    echo "[sbatch] warm-start ckpt: $CKPT"
    ;;
  phaseB)
    # Newest phaseA final checkpoint for this algo (resolved at run time —
    # the slurm dependency guarantees phaseA has finished by now).
    CKPT="$(ls -t "$OUT"/box_transport/${ALGO}/${ALGO}_*_phaseA/checkpoints/model_*.pt 2>/dev/null | head -1)"
    [ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[sbatch] phaseA ckpt NOT FOUND for $ALGO"; exit 1; }
    EXTRA_ARGS+=( "checkpoint=$CKPT" )
    echo "[sbatch] phase-B warm-start from phase-A ckpt: $CKPT"
    ;;
esac

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
