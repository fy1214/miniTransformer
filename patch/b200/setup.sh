#!/usr/bin/env bash
# =============================================================================
# B200 NVFP4 全流程工作镜像 — 一键安装脚本
#
# 适用场景:
#   - 基础镜像: slimerl/slime:v0.2.3 (B200)
#   - 训练: Megatron-LM + miniTransformer (替换 TEGroupedLinear) + TE (nvfp4)
#   - 推理: sglang (modelopt_fp4) + slime (rollout / weight update)
#
# 前置条件 (容器外):
#   docker pull slimerl/slime:v0.2.3
#   docker run --gpus all -it --rm \
#       -v /home/admin/mingfa:/root/workspace \
#       slimerl/slime:v0.2.3 bash
#
# 容器内执行:
#   cd /root && git clone <miniTransformer-repo>
#   cd /root/miniTransformer && bash patch/b200/setup.sh
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# 路径配置 — 默认匹配 slimerl/slime:v0.2.3 容器布局
# -----------------------------------------------------------------------------
: "${MINITE_DIR:=/root/miniTransformer}"
: "${MINITE_REPO:=https://git.infix-ai.xyz/infra/miniTransformer.git}"
: "${MINITE_BRANCH:=main}"
: "${MEGATRON_DIR:=/root/Megatron-LM}"
: "${SLIME_DIR:=/root/slime}"
: "${SLIME_REPO:=https://github.com/fy1214/slime.git}"
: "${SLIME_BRANCH:=nvfp4}"
# sglang in slimerl/slime:v0.2.3 container ships as an editable git checkout
# under /sgl-workspace/sglang. Override SGLANG_DIR if your layout differs.
: "${SGLANG_DIR:=/sgl-workspace/sglang}"
: "${TE_DIR:=/usr/local/lib/python3.12/dist-packages/transformer_engine}"

PATCH_ROOT="${MINITE_DIR}/patch/b200"

log()  { printf '\033[1;34m[setup.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup.sh]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[setup.sh]\033[0m %s\n' "$*" >&2; exit 1; }

require_dir() {
    local desc="$1" path="$2"
    [[ -d "$path" ]] || fail "${desc} not found at ${path} — override with \$${3}"
}

# -----------------------------------------------------------------------------
# 0. 环境自检 + 同步 miniTransformer 到 $MINITE_REPO@$MINITE_BRANCH
#    基础镜像里 /root/miniTransformer 可能停在过期分支(如 test@bce0bd9，
#    缺 QAT_PARAMS=8 等)，所以这里强制对齐到 $MINITE_BRANCH。
# -----------------------------------------------------------------------------
log "Checking environment ..."
require_dir "Megatron-LM"         "$MEGATRON_DIR" "MEGATRON_DIR"
require_dir "transformer_engine"  "$TE_DIR"       "TE_DIR"
[[ -d "$SLIME_DIR"  ]] || warn  "slime not found at $SLIME_DIR (slime patch will be skipped — override with \$SLIME_DIR)"
[[ -d "$SGLANG_DIR" ]] || warn  "sglang not found at $SGLANG_DIR (sglang patch will be skipped — override with \$SGLANG_DIR)"

command -v python3 >/dev/null || fail "python3 not in PATH"
command -v git      >/dev/null || fail "git not in PATH"

if [[ -d "$MINITE_DIR/.git" ]]; then
    minite_remote="$(git -C "$MINITE_DIR" config --get remote.origin.url 2>/dev/null || true)"
    minite_branch="$(git -C "$MINITE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ "$minite_remote" != "$MINITE_REPO" ]]; then
        log "miniTransformer remote 不是 $MINITE_REPO (当前=$minite_remote) — 移到 .bak 重新 clone"
        mv "$MINITE_DIR" "${MINITE_DIR}.orig.$(date +%s).bak"
        git clone -b "$MINITE_BRANCH" "$MINITE_REPO" "$MINITE_DIR"
    else
        log "Sync miniTransformer to $MINITE_BRANCH (was on $minite_branch)"
        git -C "$MINITE_DIR" fetch origin
        # checkout 远端分支；如果本地工作区有未提交修改会失败，提示用户处理
        git -C "$MINITE_DIR" checkout "$MINITE_BRANCH" 2>/dev/null \
            || git -C "$MINITE_DIR" checkout -t "origin/$MINITE_BRANCH"
        git -C "$MINITE_DIR" pull --ff-only origin "$MINITE_BRANCH"
    fi
elif [[ ! -e "$MINITE_DIR" ]]; then
    log "miniTransformer 不存在 — clone $MINITE_REPO@$MINITE_BRANCH 到 $MINITE_DIR"
    git clone -b "$MINITE_BRANCH" "$MINITE_REPO" "$MINITE_DIR"
else
    fail "$MINITE_DIR exists but is not a git repo; refusing to overwrite"
fi
require_dir "miniTransformer"     "$MINITE_DIR"   "MINITE_DIR"

# -----------------------------------------------------------------------------
# 1. 安装 miniTransformer
# -----------------------------------------------------------------------------
log "Step 1/6: install miniTransformer (editable, no-build-isolation)"
pushd "$MINITE_DIR" >/dev/null
git submodule update --init --recursive
pip install -e . --no-build-isolation
popd >/dev/null

# -----------------------------------------------------------------------------
# 2. 修复 nvidia-cutlass-dsl-libs-base 不完整安装
#    slimerl/slime:v0.2.3 基础镜像里这个 wheel 的 RECORD 列了 340 个文件，
#    但磁盘上只剩 44 个 — 缺 .pth 钩子、cutlass/__init__.py、_cutlass_ir.so
#    等关键文件，导致 `import cutlass` 失败。强制重装可一并补齐。
# -----------------------------------------------------------------------------
log "Step 2/6: force-reinstall nvidia-cutlass-dsl-libs-base (fix incomplete base image install)"
if python3 -c "import cutlass, cutlass.torch, cutlass.cute" 2>/dev/null; then
    log "  [skip] cutlass / cutlass.torch / cutlass.cute already importable"
else
    pip install --force-reinstall --no-deps nvidia-cutlass-dsl-libs-base==4.4.2
    python3 -c "import cutlass, cutlass.torch, cutlass.cute" \
        || fail "cutlass still not importable after reinstall"
    log "  [done] cutlass importable"
fi

# -----------------------------------------------------------------------------
# helper: apply a unified-diff patch from PATCH_ROOT to a target directory
#         idempotent — uses `git apply --check` first, then `--reverse` probe
# -----------------------------------------------------------------------------
apply_git_patch() {
    local target_dir="$1" patch_file="$2"
    [[ -f "$patch_file" ]] || fail "patch not found: $patch_file"

    pushd "$target_dir" >/dev/null
    if git apply --check --reverse "$patch_file" 2>/dev/null; then
        log "  [skip] already applied: $(basename "$patch_file")"
    elif git apply --check "$patch_file" 2>/dev/null; then
        git apply "$patch_file"
        log "  [done] applied: $(basename "$patch_file")"
    else
        # last resort: try plain `patch -p1 --forward` with whitespace tolerance
        if patch -p1 --forward --dry-run --silent -i "$patch_file" 2>/dev/null; then
            patch -p1 --forward -i "$patch_file"
            log "  [done] applied via patch(1): $(basename "$patch_file")"
        else
            fail "cannot apply $(basename "$patch_file") to $target_dir — file already modified differently or version mismatch"
        fi
    fi
    popd >/dev/null
}

# -----------------------------------------------------------------------------
# 3. Patch sglang (NVFP4 per-token scale)
# -----------------------------------------------------------------------------
if [[ -d "$SGLANG_DIR" ]]; then
    log "Step 3/6: apply sglang patch(es)"
    # Patches use `a/python/sglang/...` paths, so they apply at the sglang repo
    # root (one level above python/sglang/).  In slimerl/slime:v0.2.3 the repo
    # root IS $SGLANG_DIR (/sgl-workspace/sglang) and has a .git dir.  For a
    # site-package layout (no .git), we fall back to `patch -p2` against the
    # parent dir.
    sglang_root="$SGLANG_DIR"
    for p in "${PATCH_ROOT}/sglang/"*.patch; do
        [[ -f "$p" ]] || continue
        log "  -> $(basename "$p")"
        if [[ -d "${sglang_root}/.git" ]]; then
            apply_git_patch "$sglang_root" "$p"
        else
            pushd "$sglang_root/.." >/dev/null
            if patch --dry-run -R -p2 --silent -i "$p" 2>/dev/null; then
                log "  [skip] already applied: $(basename "$p")"
            elif patch --dry-run -p2 --forward --silent -i "$p" 2>/dev/null; then
                patch -p2 --forward -i "$p"
                log "  [done] applied via patch -p2: $(basename "$p")"
            else
                warn "  could not auto-apply $(basename "$p") to $sglang_root — please inspect manually"
            fi
            popd >/dev/null
        fi
    done
else
    log "Step 3/6: SGLANG_DIR not present — skipping sglang patch"
fi

# -----------------------------------------------------------------------------
# 4. Patch Megatron-LM (Fp8Padding align_size=128 + TEGroupedLinear override)
# -----------------------------------------------------------------------------
log "Step 4/6: apply Megatron-LM patch"
for p in "${PATCH_ROOT}/megatron/"*.patch; do
    [[ -f "$p" ]] || continue
    log "  -> $(basename "$p")"
    apply_git_patch "$MEGATRON_DIR" "$p"
done

# -----------------------------------------------------------------------------
# 5. Patch TransformerEngine (disable FP8 fallback for unreplaced Linear)
# -----------------------------------------------------------------------------
log "Step 5/6: apply TransformerEngine patch (Python in-place edit)"
python3 "${PATCH_ROOT}/transformer-engine/apply_disable_fp8.py" --te-dir "$TE_DIR"

# -----------------------------------------------------------------------------
# 6. Patch slime (force post_process_weights after every update)
# -----------------------------------------------------------------------------
# slimerl/slime:v0.2.3 ships THUDM/slime main, but our slime patch is written
# against fy1214/slime nvfp4 (which adds NVFP4-aware weight-update plumbing).
# If $SLIME_DIR doesn't exist OR isn't the right branch, clone it fresh.
if [[ -d "$SLIME_DIR/.git" ]]; then
    slime_branch="$(git -C "$SLIME_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    slime_remote="$(git -C "$SLIME_DIR" config --get remote.origin.url 2>/dev/null || true)"
    if [[ "$slime_branch" != "$SLIME_BRANCH" || "$slime_remote" != "$SLIME_REPO" ]]; then
        log "Step 6/6: existing $SLIME_DIR is $slime_remote@$slime_branch — moving aside and re-cloning $SLIME_REPO@$SLIME_BRANCH"
        mv "$SLIME_DIR" "${SLIME_DIR}.orig.$(date +%s).bak"
        git clone --depth 1 -b "$SLIME_BRANCH" "$SLIME_REPO" "$SLIME_DIR"
    fi
elif [[ ! -e "$SLIME_DIR" ]]; then
    log "Step 6/6: $SLIME_DIR does not exist — cloning $SLIME_REPO@$SLIME_BRANCH"
    git clone --depth 1 -b "$SLIME_BRANCH" "$SLIME_REPO" "$SLIME_DIR"
fi

if [[ -d "$SLIME_DIR" ]]; then
    log "Step 6/6: apply slime patch"
    for p in "${PATCH_ROOT}/slime/"*.patch; do
        [[ -f "$p" ]] || continue
        log "  -> $(basename "$p")"
        apply_git_patch "$SLIME_DIR" "$p"
    done
else
    log "Step 6/6: SLIME_DIR not present — skipping slime patch"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat <<'EOF'

==============================================================================
  Setup complete.  Next steps:
==============================================================================

  1) Quantize a HF model to NVFP4 (expert-only) — run from slime repo on the
     nvfp4 branch (https://github.com/fy1214/slime.git -b nvfp4):

       python tools/convert_hf_to_nvfp4.py \
           --model-dir  /path/to/hf/Qwen3-30B-A3B \
           --save-dir   /path/to/hf/Qwen3-30B-A3B-NVFP4

  2) Required training/inference args (see CLAUDE.md for full context):

     SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 1
        --sglang-mem-fraction-static 0.7
        --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
        --sglang-quantization modelopt_fp4
     )
     PRECISE_ARGS=(
        --transformer-impl transformer_engine
        --bf16
        --fp4-format e2m1
        --fp4-recipe nvfp4
     )

  3) Required env vars:

     PYTHONPATH=/root/Megatron-LM:/root/miniTransformer
     CUDA_DEVICE_MAX_CONNECTIONS=1
     NVTE_NVFP4_DISABLE_RHT=0                 # enable RHT in backward
     NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 # enable stochastic rounding in quantize
     NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1     # 1D (matches sglang modelopt_fp4 inference)
     QAT_PARAM=8

  4) To redo only one step, set SKIP_* env vars or call the patch directly:
       python3 patch/b200/transformer-engine/apply_disable_fp8.py
       (cd $MEGATRON_DIR && git apply patch/b200/megatron/*.patch)
       (cd $SLIME_DIR    && git apply patch/b200/slime/*.patch)

==============================================================================
EOF
