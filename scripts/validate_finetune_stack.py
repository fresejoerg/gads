"""Toolchain validation for the GADS fine-tuning loop.

Ordered cheapest-first so a failure names its own cause: raw CUDA kernels for this GPU,
then 4-bit quantization, then the actual Unsloth LoRA path. The RTX 5080 Laptop is
Blackwell (sm_120); wheels built before that support exists fail HERE rather than looking
like a data problem later.

    .venv-train/bin/python scripts/validate_finetune_stack.py         # cheap checks only
    .venv-train/bin/python scripts/validate_finetune_stack.py --full  # + a real LoRA step

Environment build (validated 2026-09-03, 5/5 pass on an RTX 5080 Laptop). Two non-obvious
constraints, both of which fail LATER and misleadingly if got wrong:

    uv python install 3.12
    uv venv .venv-train --python ~/.local/share/uv/python/cpython-3.12*/bin/python3.12
    VIRTUAL_ENV=.venv-train uv pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    VIRTUAL_ENV=.venv-train uv pip install unsloth numpy

1. **Install torchvision from the SAME cu128 index as torch.** `pip install unsloth` pulls
   a default-index torchvision built against CUDA 13.0, which raises
   "PyTorch and torchvision were compiled with different CUDA major versions" on import —
   surfacing as an unsloth import failure rather than a dependency one.
2. **Use a uv-managed Python, not the system one.** Triton JIT-compiles a C shim at first
   use and needs `Python.h`; Ubuntu's python3.12 ships no dev headers, so the LoRA step
   dies in gcc. uv's standalone builds include headers, which avoids needing sudo.

Kept deliberately separate from the GADS runtime venv (`.venv`): the training stack is
~6.5 GB and must not enter the server's dependency set.
"""
import sys

ok, fail = [], []


def check(name, fn):
    try:
        detail = fn()
        ok.append(name)
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        fail.append(name)
        print(f"  FAIL  {name} — {type(e).__name__}: {str(e)[:160]}")
        return False


print("\n1. torch + this GPU")


def _torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available to torch")
    cap = torch.cuda.get_device_capability()
    free, total = torch.cuda.mem_get_info()
    globals()["_free_gb"] = free / 1e9
    return (f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
            f"sm_{cap[0]}{cap[1]} | {free/1e9:.1f}/{total/1e9:.1f} GB free")


if not check("torch sees the GPU", _torch):
    sys.exit("\nStop: nothing below can pass.")


def _kernels():
    import torch
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    (a @ b).sum().backward()
    torch.cuda.synchronize()
    if a.grad is None:
        raise RuntimeError("no gradient produced")
    return "bf16 matmul + backward on device"


check("sm_120 kernels actually run", _kernels)

print("\n2. 4-bit quantization (bitsandbytes)")


def _bnb():
    import torch, bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.bfloat16).cuda()
    out = lin(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    return f"bitsandbytes {bnb.__version__} | Linear4bit forward {tuple(out.shape)}"


check("Linear4bit forward on this GPU", _bnb)

print("\n3. library imports")


def _imports():
    import unsloth, transformers, trl, peft
    return (f"unsloth {unsloth.__version__} | transformers {transformers.__version__} | "
            f"trl {trl.__version__} | peft {peft.__version__}")


check("unsloth / transformers / trl / peft", _imports)

if "--full" not in sys.argv:
    print(f"\n{len(ok)} passed, {len(fail)} failed. "
          f"Re-run with --full for a real LoRA step (needs free VRAM).")
    sys.exit(1 if fail else 0)

print("\n4. real LoRA training step")
free = globals().get("_free_gb", 0)
if free < 4:
    print(f"  SKIP  only {free:.1f} GB VRAM free — unload the model in LM Studio first.")
    sys.exit(1 if fail else 0)


def _lora():
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    model, tok = FastLanguageModel.from_pretrained(
        # Deliberately tiny: this validates the PATH, not the model.
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=1024, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=8, lora_alpha=16, use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    ds = Dataset.from_list([
        {"text": f"### Task\nsum a list\n### Code\ndef f(x): return sum(x)  # {i}"}
        for i in range(8)])
    trainer = SFTTrainer(
        model=model, train_dataset=ds, processing_class=tok,
        args=SFTConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1,
                       max_steps=2, learning_rate=2e-4, logging_steps=1,
                       output_dir="/tmp/unsloth_validate", report_to="none",
                       max_length=1024, completion_only_loss=False))
    stats = trainer.train()
    return f"2 steps, final loss {stats.training_loss:.4f}"


check("Unsloth LoRA: load 4-bit -> attach adapter -> 2 steps", _lora)
print(f"\n{len(ok)} passed, {len(fail)} failed.")
sys.exit(1 if fail else 0)
