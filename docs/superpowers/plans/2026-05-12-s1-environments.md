# S1: Environment Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get two conda environments working on local and (once provisioned) remote GPU hosts: `unifyptq` (Py 3.10 + torch 2.6 + torchao) for FlatQuant, CompSRT, KV-PCSA; and `ahcptq-old` (Py 3.9 + torch 1.13 + mmcv-full 1.7 + vendored mmdet 2.x) for paper-faithful AHCPTQ reproduction. Set up rsync sync between machines and a shared checkpoint directory layout.

**Architecture:** Two independent conda envs per machine. Shared `/home/ubuntu/unifying-ptq/results/` layout. rsync over SSH (no NFS) for sync. SSH key-based auth so unattended commands work.

**Tech Stack:** miniconda3, conda, pip, PyTorch, torchao, mmcv-full, openmim, rsync, ssh.

---

### Task S1.1: Verify local `unifyptq` env

**Files:**
- Read: `/home/ubuntu/miniconda3/envs/unifyptq/` (already exists)

- [ ] **Step 1: Activate env and confirm torch + CUDA work**

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.6.0+cu124 True NVIDIA A100 80GB PCIe`.

- [ ] **Step 2: Confirm FlatQuant imports clean**

```bash
cd /home/ubuntu/unifying-ptq/FlatQuant && python -c "import flatquant.utils, flatquant.args_utils, flatquant.model_utils, flatquant.data_utils, flatquant.eval_utils, flatquant.train_utils, flatquant.flat_utils, flatquant.hadamard_utils; print('FQ_OK')"
```

Expected: `FQ_OK`.

- [ ] **Step 3: Confirm CompSRT imports clean**

```bash
cd /home/ubuntu/unifying-ptq/CompSRT && python -c "import basicsr; print('BSR_OK')"
```

Expected: `BSR_OK`.

---

### Task S1.2: Install torchao in `unifyptq`

**Files:**
- Modify: pip site-packages in `unifyptq` env

- [ ] **Step 1: Install torchao matching torch 2.6**

```bash
conda activate unifyptq
pip install torchao 2>&1 | tail -5
```

Expected: `Successfully installed torchao-X.Y.Z` (no torch reinstall).

- [ ] **Step 2: Verify torchao W4A16 path works**

```bash
python -c "
import torch
from torchao.quantization import quantize_, Int4WeightOnlyConfig
m = torch.nn.Linear(256, 256).cuda().half()
quantize_(m, Int4WeightOnlyConfig())
x = torch.randn(4, 256, device='cuda', dtype=torch.half)
print('torchao_W4A16_OK', m(x).shape)
"
```

Expected: `torchao_W4A16_OK torch.Size([4, 256])`.

- [ ] **Step 3: Verify torchao W4A4 path imports**

```bash
python -c "
from torchao.quantization import Int4DynamicActivationInt4WeightConfig
print('W4A4_CONFIG_OK')
"
```

Expected: `W4A4_CONFIG_OK` (it's fine if running on a matmul fails later — we just need the config to be importable now).

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/unifying-ptq
pip freeze | grep -E '^torchao' > /tmp/torchao_version.txt
git diff --stat
# No code changes; record the env state in docs
echo "$(date -u +%Y-%m-%d) torchao installed: $(cat /tmp/torchao_version.txt)" >> docs/superpowers/specs/env-log.md
git add docs/superpowers/specs/env-log.md
git commit -m "feat(env): install torchao in unifyptq for real INT4 deployment"
```

---

### Task S1.3: Build `ahcptq-old` env (paper-faithful AHCPTQ stack)

**Files:**
- Create: `/home/ubuntu/miniconda3/envs/ahcptq-old/`
- Modify: `docs/superpowers/specs/env-log.md`

- [ ] **Step 1: Create env with Py 3.9**

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda create -n ahcptq-old python=3.9 -y
conda activate ahcptq-old
```

Expected: env created, prompt shows `(ahcptq-old)`.

- [ ] **Step 2: Install torch 1.13 + cu117**

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117 2>&1 | tail -3
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `1.13.1+cu117 True`.

- [ ] **Step 3: Install mmcv-full 1.7 via mim**

```bash
pip install -U "openmim==0.3.7"
mim install "mmcv-full==1.7.0" 2>&1 | tail -5
python -c "import mmcv; print('MMCV', mmcv.__version__)"
```

Expected: `MMCV 1.7.0`. If mim refuses, fall back to:

```bash
pip install "mmcv-full==1.7.0" -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html 2>&1 | tail -5
```

- [ ] **Step 4: Install AHCPTQ runtime requirements**

```bash
cd /home/ubuntu/unifying-ptq
pip install -r requirements.txt 2>&1 | tail -5
```

Expected: all packages install (matplotlib, numpy<2, pycocotools, scipy, six, terminaltables, `timm==0.9.8`, easydict, etc.).

- [ ] **Step 5: Install vendored mmdetection 2.x**

```bash
cd /home/ubuntu/unifying-ptq/mmdetection
python setup.py build develop 2>&1 | tail -5
python -c "import mmdet; print('MMDET', mmdet.__version__)"
```

Expected: `MMDET 2.x.y` (the vendored version).

- [ ] **Step 6: Compile AHCPTQ CUDA ops**

```bash
cd /home/ubuntu/unifying-ptq/projects/instance_segment_anything/ops
python setup.py build install 2>&1 | tail -5
cd /home/ubuntu/unifying-ptq
python -c "from projects.instance_segment_anything.ops.modules.ms_deform_attn import MSDeformAttn; print('MSDA_OK')"
```

Expected: `MSDA_OK`.

- [ ] **Step 7: Verify AHCPTQ imports end-to-end**

```bash
python -c "
import sys; sys.path.insert(0, '/home/ubuntu/unifying-ptq')
import ahcptq.solver.test_quant as t
print('AHCPTQ_OK')
"
```

Expected: `AHCPTQ_OK` (no traceback).

- [ ] **Step 8: Commit env log**

```bash
echo "$(date -u +%Y-%m-%d) ahcptq-old env built: torch 1.13.1+cu117, mmcv-full 1.7.0, mmdet vendored" >> docs/superpowers/specs/env-log.md
git add docs/superpowers/specs/env-log.md
git commit -m "feat(env): build ahcptq-old env for paper-faithful SAM reproduction"
```

---

### Task S1.4: Provision remote A100 + bootstrap SSH

**Files:**
- Create: `/home/ubuntu/.ssh/remote_gpu` (private key)
- Modify: `/home/ubuntu/.ssh/config`

**Prerequisite:** User has rented an A100 80GB on Lambda Labs, RunPod, or Vast.ai and shared the host + port + key. Mark blocked until they do.

- [ ] **Step 1: Store SSH key + config**

```bash
mkdir -p /home/ubuntu/.ssh
# User pastes the private key contents — save to /home/ubuntu/.ssh/remote_gpu via heredoc when user provides
chmod 600 /home/ubuntu/.ssh/remote_gpu
cat >> /home/ubuntu/.ssh/config <<'EOF'
Host remote-gpu
  HostName <USER_PROVIDED_HOST>
  Port <USER_PROVIDED_PORT>
  User <USER_PROVIDED_USER>
  IdentityFile /home/ubuntu/.ssh/remote_gpu
  StrictHostKeyChecking accept-new
EOF
chmod 600 /home/ubuntu/.ssh/config
```

- [ ] **Step 2: Verify connectivity**

```bash
ssh remote-gpu 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && uname -a'
```

Expected: prints `NVIDIA A100 80GB PCIe, 81920 MiB` (or similar) and Linux kernel info.

- [ ] **Step 3: Install miniconda on remote (idempotent)**

```bash
ssh remote-gpu 'bash -lc "
if [ ! -d \$HOME/miniconda3 ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p \$HOME/miniconda3
  \$HOME/miniconda3/bin/conda init bash
  \$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  \$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi
\$HOME/miniconda3/bin/conda --version
"'
```

Expected: `conda 24.x.y` (or current version).

- [ ] **Step 4: Install rsync on remote (and locally if missing)**

```bash
ssh remote-gpu 'which rsync || (apt-get update -qq && apt-get install -y -qq rsync)'
which rsync || sudo apt-get install -y -qq rsync
```

Expected: both prints `/usr/bin/rsync`.

- [ ] **Step 5: Clone repo on remote**

```bash
ssh remote-gpu 'cd ~ && git clone https://github.com/anomous-researcher-gldz/unifying-ptq.git || (cd unifying-ptq && git pull)'
ssh remote-gpu 'ls ~/unifying-ptq/FlatQuant ~/unifying-ptq/CompSRT ~/unifying-ptq/ahcptq | head'
```

Expected: directory listing matching local.

- [ ] **Step 6: Build remote `unifyptq` env (mirror local)**

Reuse the install steps from S1.1–S1.2 but on remote:

```bash
ssh remote-gpu 'bash -lc "
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n unifyptq python=3.10 -y
conda activate unifyptq
pip install --upgrade pip wheel setuptools
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torchao
pip install transformers==4.45.0 accelerate==0.32.0 datasets==2.17.1 lm-eval==0.4.9.1 termcolor 'triton>=3.2.0' matplotlib brokenaxes sentencepiece protobuf
pip install ninja packaging
"'
```

Expected: torchao + deps installed without torch downgrade. Verify:

```bash
ssh remote-gpu 'bash -lc "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python -c 'import torch, torchao; print(torch.__version__, torch.cuda.is_available(), torchao.__version__)'
"'
```

Expected: `2.6.0+cu124 True <torchao version>`.

- [ ] **Step 7: Build remote `ahcptq-old` env (mirror local Task S1.3)**

Repeat the entire S1.3 sequence over SSH (env create, torch 1.13, mmcv-full 1.7, requirements, mmdetection, ops). Or copy the local env via `conda env export` + `conda env create -f` if faster.

Expected: `ssh remote-gpu 'conda activate ahcptq-old && python -c "import mmcv, mmdet; print(mmcv.__version__, mmdet.__version__)"'` prints the same versions as local.

- [ ] **Step 8: Commit log**

```bash
echo "$(date -u +%Y-%m-%d) remote-gpu provisioned: <host>, miniconda + unifyptq + ahcptq-old" >> docs/superpowers/specs/env-log.md
git add docs/superpowers/specs/env-log.md
git commit -m "feat(env): provision remote A100 with unifyptq + ahcptq-old envs"
```

---

### Task S1.5: rsync wrapper for cross-machine sync

**Files:**
- Create: `/home/ubuntu/unifying-ptq/scripts/sync_results.sh`

- [ ] **Step 1: Write the sync script**

```bash
mkdir -p /home/ubuntu/unifying-ptq/scripts
cat > /home/ubuntu/unifying-ptq/scripts/sync_results.sh <<'EOF'
#!/usr/bin/env bash
# Two-way rsync of results/ between local and remote-gpu.
# Usage: ./sync_results.sh push | pull | both
set -euo pipefail
DIR="results"
REMOTE_DIR="~/unifying-ptq/$DIR"
LOCAL_DIR="/home/ubuntu/unifying-ptq/$DIR"
mkdir -p "$LOCAL_DIR"
ssh remote-gpu "mkdir -p $REMOTE_DIR"
case "${1:-both}" in
  push) rsync -avz --partial "$LOCAL_DIR/" "remote-gpu:$REMOTE_DIR/" ;;
  pull) rsync -avz --partial "remote-gpu:$REMOTE_DIR/" "$LOCAL_DIR/" ;;
  both) rsync -avz --partial "$LOCAL_DIR/" "remote-gpu:$REMOTE_DIR/" && rsync -avz --partial "remote-gpu:$REMOTE_DIR/" "$LOCAL_DIR/" ;;
  *) echo "usage: $0 push|pull|both" >&2; exit 1 ;;
esac
EOF
chmod +x /home/ubuntu/unifying-ptq/scripts/sync_results.sh
```

- [ ] **Step 2: Test the script (dry run with empty dir)**

```bash
mkdir -p /home/ubuntu/unifying-ptq/results
touch /home/ubuntu/unifying-ptq/results/.placeholder
/home/ubuntu/unifying-ptq/scripts/sync_results.sh push
ssh remote-gpu 'ls ~/unifying-ptq/results/'
```

Expected: prints `.placeholder` from remote.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/sync_results.sh
git commit -m "feat(infra): rsync wrapper for cross-machine results sync"
```

---

### Task S1.6: Create checkpoint directory layout

**Files:**
- Create: `/home/ubuntu/unifying-ptq/results/` subtree

- [ ] **Step 1: Layout directories**

```bash
cd /home/ubuntu/unifying-ptq
mkdir -p results/{S2-ahcptq,S4-dbaf-weak,S5-kv-pcsa,S6-int4,S7-ablations,S8-compsrt,S9-downstream,deploy}
touch results/.gitkeep
cat > results/README.md <<'EOF'
# results/

Per-experiment outputs. Layout:

```
results/
├── S2-ahcptq/<model>/<method>/<seed>/{state.pt,eval.json,logs/}
├── S4-dbaf-weak/<model>/<baseline>/<seed>/{state.pt,eval.json}
├── S5-kv-pcsa/<model>/<context_len>/<seed>/{state.pt,eval.json}
├── S6-int4/<model>/<backend>/{packed_weights/,bench.json}
├── S7-ablations/<exp>/<model>/<seed>/{state.pt,eval.json}
├── S8-compsrt/<model>/<method>/<seed>/{state.pt,eval.json}
├── S9-downstream/<task>/<model>/<seed>/eval.json
└── deploy/<model>/<format>/{config.json,model.safetensors}
```

eval.json schema: `{model, bits, method, task, metric, value, seed, timestamp}`.
EOF
```

- [ ] **Step 2: Sync to remote**

```bash
./scripts/sync_results.sh push
```

- [ ] **Step 3: Commit**

```bash
git add results/.gitkeep results/README.md
git commit -m "feat(infra): checkpoint directory layout for all experiments"
```

---

## Done when

- `conda activate unifyptq && python -c "import torch, torchao; print(torch.cuda.is_available())"` prints `True` on both local and remote
- `conda activate ahcptq-old && python -c "import mmcv, mmdet"` succeeds on both local and remote
- `./scripts/sync_results.sh push` and `pull` both succeed
- `results/` directory tree exists on both machines
- All commits land on `main`
