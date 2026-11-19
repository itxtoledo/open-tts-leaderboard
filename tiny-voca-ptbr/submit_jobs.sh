#!/bin/bash
# HF Jobs TTS eval — tiny-voca-ptbr backend (fixed voice → generate → transcribe → score; no SIM, and UTMOS is English-only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash tiny-voca-ptbr/submit_jobs.sh ["semfence/tiny-voca-ptbr F_s4" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 ONLY_LANGS="pt" bash tiny-voca-ptbr/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh.
source "${BACKEND_DIR}/job_image.sh"
# torch intra-op threads (run_eval.py's --threads). 1: the model's matrices are small, and a fixed
# count keeps RTFx independent of the h200 host's core count.
TINY_VOCA_PTBR_THREADS="${TINY_VOCA_PTBR_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); the utmos stage skips non-English splits
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice → output suffix ────────────────────────────────────────────────────
# Both voices share one repo, so a non-default voice is a leaderboard variant: MODE_SUFFIX must match
# the suffix run_eval.py writes (publish with `open_results_pr.py --variant msm4`).
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    case "${VOICE:-F_s4}" in
        F_s4) MODE_SUFFIX="" ;;
        M_sm4) MODE_SUFFIX="_msm4" ;;
        *) echo "tiny-voca-ptbr: unknown voice '${VOICE}' (F_s4, M_sm4)" >&2; exit 1 ;;
    esac
    true
}

# Brazilian Portuguese only (pt-BR phonemizer). MiniMax's `pt` split is European Portuguese, so
# this row reads pt-PT text with a pt-BR voice, like Kokoro's `pf_dora` (Portuguese-BR).
supports_language() { [[ "$1" == "pt" ]]; }

# ── Stage 1: TTS generation ─────────────────────────────────────────────────
# CPU on the host CPU of the h200 flavor; --device=cpu is enforced inside run_eval.py.
generate_stage() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cpu)
            cd /app/tiny-voca-ptbr && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-F_s4} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${TINY_VOCA_PTBR_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" ────────────────────
DATASET_CONFIGS=(
    "tts pt pt ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id voice" (CLI args override) ────────────────────────────
MODEL_CONFIGS=(
    "semfence/tiny-voca-ptbr F_s4"
    "semfence/tiny-voca-ptbr M_sm4"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
