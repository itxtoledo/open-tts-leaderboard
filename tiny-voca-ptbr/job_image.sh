# Stage-1 environment for the tiny-voca-ptbr backend, sourced by tiny-voca-ptbr/submit_jobs.sh.
# Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-ptbr is a 4.5 M PyTorch model that runs on CPU (code Apache-2.0,
# weights CC BY 4.0), so the job uses the public slim Python image and pip-installs the pinned
# packages at job start (~15 s). The weights come from the model repo at the pinned revision set in
# run_eval.py.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# tiny-voca-ptbr and its phonemizer (dizione-ptbr, which ships its own lexicon) are GitHub-only
# Python packages (no PyPI release yet), pinned to their 2026-11-19 main HEADs. Torch is the CPU
# wheel from PyTorch's /whl/cpu index.
TINY_VOCA_PTBR_REV="${TINY_VOCA_PTBR_REV:-2f1531068cf69e105dc9a3fac5cf35fb4dd51bc2}"
DIZIONE_PTBR_REV="${DIZIONE_PTBR_REV:-6a3baa3432a16ca84ec67529e47399aa31866f39}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# python:3.12-slim ships without git, which pip needs for the GitHub installs. Unlike the English
# tiny-voca, this front end needs no eSpeak. Then torch from the CPU wheel index (so +cpu is
# resolvable), and the rest from PyPI (so --index-url doesn't apply to packages that only live there).
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git >/dev/null &&
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    git+https://github.com/SemFence/dizione-ptbr@${DIZIONE_PTBR_REV} \
    git+https://github.com/SemFence/tiny-voca-ptbr@${TINY_VOCA_PTBR_REV} &&
INSTALL
}
