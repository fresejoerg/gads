"""QLoRA fine-tune of the local engine on harvested GADS traces (approach_docs/031 §7 P3).

Run it from the TRAINING venv, not the GADS one:

    .venv-train/bin/python scripts/train_lora.py \\
        --model unsloth/gemma-3-12b-it-unsloth-bnb-4bit \\
        --data research/finetune/sft_train.jsonl \\
        --eval research/finetune/sft_holdout.jsonl \\
        --run-name gemma12b-sft-v1

Everything is logged to MLflow (default http://localhost:5000, experiment `gads-distill`):
loss curves live via the HF callback, plus the provenance that makes a checkpoint
*interpretable* six months later — which corpus, which prompt regime, which serving engine
produced the traces, and which specs were held out. A checkpoint whose training data cannot
be identified is not usable evidence, and this is the file that prevents that.

Defaults follow the measurements in 031 §6, not habit:

  max_seq_length 8192  — p95 of real Coder prompts is 10,653 tokens; 2048 (the library
                         default) would truncate or discard 93% of the corpus.
  train_on_completions — mask the prompt. Otherwise the model is trained to reproduce file
                         schemas and kernel snapshots, which it is never asked to produce.
  4-bit QLoRA          — 16 GB card, and serving must fit alongside nothing else.

**Training and serving contend for the same GPU.** Unload the model in LM Studio first, or
this OOMs. GADS cannot run local workflows while this is training.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def sha_of(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def corpus_provenance(path):
    """What this corpus is made of — the facts a future reader needs to trust a checkpoint."""
    rows = [json.loads(l) for l in open(path)]
    specs = sorted({r.get("spec") for r in rows if r.get("spec")})
    prov = {"n_examples": len(rows), "n_specs": len(specs),
            "corpus_sha256": sha_of(path)}
    # The manifest carries per-example provenance the training file does not.
    man = os.path.join(os.path.dirname(path), "manifest.jsonl")
    if os.path.exists(man):
        m = [json.loads(l) for l in open(man)]
        import collections
        prov["source_models"] = json.dumps(
            dict(collections.Counter(x.get("model_used") for x in m).most_common()))
        prov["source_engines"] = json.dumps(
            dict(collections.Counter(x.get("engine_id") or "unstamped" for x in m)))
        prov["prompt_versions"] = json.dumps(
            dict(collections.Counter(x.get("prompt_version") or "?" for x in m)))
    return prov, specs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF base checkpoint (4-bit ok)")
    ap.add_argument("--data", default="research/finetune/sft_train.jsonl")
    ap.add_argument("--eval", default="research/finetune/sft_holdout.jsonl")
    ap.add_argument("--out", default="research/finetune/adapters")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--experiment", default="gads-distill")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI",
                                                        "http://localhost:5000"))
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-completion-only", action="store_true",
                    help="train on the whole sequence (NOT recommended — see module docs)")
    args = ap.parse_args()

    run_name = args.run_name or f"lora-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir = os.path.join(args.out, run_name)

    # MLflow env must be set BEFORE the HF Trainer builds its callback list, which is what
    # streams the live loss curve; setting it afterwards silently logs nothing.
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment
    os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "true")

    import unsloth  # noqa: F401  — must precede transformers to apply its patches
    import mlflow
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    prov, specs = corpus_provenance(args.data)
    _, held_specs = corpus_provenance(args.eval) if os.path.exists(args.eval) else ({}, [])

    print(f"corpus: {prov['n_examples']} examples / {prov['n_specs']} specs "
          f"(sha {prov['corpus_sha256']})")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)} | {free/1e9:.1f}/{total/1e9:.1f} GB free")
    if free / 1e9 < 6:
        print("  ! Little free VRAM — unload the model in LM Studio first.", flush=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "gads.git_rev": git_rev(),
            "gads.stage": "coder-sft",
            "gads.base_model": args.model,
            "gads.gpu": torch.cuda.get_device_name(0),
            # Which specs the model must NOT have seen. Recorded as a tag so a later
            # evaluation cannot silently score on training data.
            "gads.holdout_specs": ",".join(held_specs)[:4900],
        })
        mlflow.log_params({
            **prov,
            "base_model": args.model, "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout, "lr": args.lr, "epochs": args.epochs,
            "max_steps": args.max_steps, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "seed": args.seed,
            "train_on_completions": not args.no_completion_only,
            "load_in_4bit": True,
        })
        for f in (args.data, args.eval,
                  os.path.join(os.path.dirname(args.data), "manifest.jsonl"),
                  "research/finetune/eval_report.jsonl"):
            # The pre-training baseline travels WITH the run: a post-training number is
            # meaningless without the number it is being compared against.
            if os.path.exists(f):
                try:
                    mlflow.log_artifact(f, artifact_path="corpus")
                except Exception as e:
                    print(f"  ! could not log {f}: {type(e).__name__}")

        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_seq_length,
            load_in_4bit=True, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", random_state=args.seed,
            use_gradient_checkpointing="unsloth",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

        def to_prompt_completion(split_file):
            """Split `messages` into TRL's prompt/completion columns.

            A single `messages` list gives TRL no boundary to mask at, so
            `completion_only_loss` has nothing to act on (and unsloth rejects the shape
            outright). Prompt/completion states the boundary explicitly, which is what
            makes "train on the assistant turn only" actually true rather than assumed.
            The harvester's task_id/spec/required_variables columns are dropped here —
            they exist for eval provenance, not for the trainer.
            """
            d = load_dataset("json", data_files={"s": split_file})["s"]
            return d.map(
                lambda r: {"prompt": r["messages"][:-1],
                           "completion": r["messages"][-1:]},
                remove_columns=d.column_names)

        ds = to_prompt_completion(args.data)
        eval_ds = to_prompt_completion(args.eval) if os.path.exists(args.eval) else None

        # PRE-FLIGHT: a prompt longer than max_seq_length truncates the assistant turn
        # away entirely, leaving zero unmasked labels — the loss is then NaN and the run
        # burns to completion having learned nothing. Observed directly: identical data at
        # 2048 gives loss=nan/grad_norm=0, at 8192 gives 1.06 -> 0.87. Silent, so it is
        # checked rather than trusted.
        def _prompt_len(rec):
            return len(tok.apply_chat_template(rec["prompt"], tokenize=True,
                                               add_generation_prompt=True))
        probe = ds.select(range(min(len(ds), 64)))
        lens = [_prompt_len(r) for r in probe]
        starved = sum(1 for L in lens if L >= args.max_seq_length)
        pct = 100.0 * starved / max(len(lens), 1)
        print(f"pre-flight: prompt tokens over {len(lens)} sampled examples — "
              f"median {sorted(lens)[len(lens)//2]}, max {max(lens)}; "
              f"{starved} ({pct:.0f}%) leave no room for the completion")
        mlflow.log_metrics({"preflight_prompt_tokens_max": float(max(lens)),
                            "preflight_starved_pct": pct})
        if pct > 50:
            sys.exit(f"Refusing to train: {pct:.0f}% of examples would train on zero "
                     f"completion tokens at max_seq_length={args.max_seq_length}. "
                     f"Raise it (p95 of real Coder prompts is 10,653) or re-harvest with "
                     f"a smaller --max-tokens.")
        if starved:
            print(f"  ! {starved} example(s) will contribute no loss signal.", flush=True)

        cfg = SFTConfig(
            output_dir=outdir, run_name=run_name, report_to="mlflow",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, warmup_ratio=args.warmup_ratio,
            logging_steps=1, save_strategy="epoch", seed=args.seed,
            optim="adamw_8bit", lr_scheduler_type="cosine",
            bf16=True, max_length=args.max_seq_length,
            # Loss on the assistant turn only — the whole point (see module docs).
            completion_only_loss=not args.no_completion_only,
            eval_strategy="epoch" if eval_ds is not None else "no",
            per_device_eval_batch_size=1,
        )
        trainer = SFTTrainer(model=model, train_dataset=ds, eval_dataset=eval_ds,
                             processing_class=tok, args=cfg)

        print(f"\ntraining -> {outdir}\nMLflow: {args.tracking_uri} "
              f"| experiment={args.experiment} | run={run_name}", flush=True)
        stats = trainer.train()

        model.save_pretrained(outdir)
        tok.save_pretrained(outdir)
        mlflow.log_metrics({
            "final_train_loss": float(stats.training_loss),
            "train_runtime_s": float(stats.metrics.get("train_runtime", 0)),
            "peak_vram_gb": torch.cuda.max_memory_reserved() / 1e9,
        })
        mlflow.log_artifact(os.path.join(outdir, "adapter_config.json"),
                            artifact_path="adapter")
        print(f"\ndone. adapter -> {outdir}")
        print(f"MLflow run: {args.tracking_uri}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")
        print("\nNext: this checkpoint is a DISTINCT engine. Serve it with "
              "GADS_ENGINE_TAG set so its ledger rows can never be read as baseline "
              "(approach_docs/031 §5).")


if __name__ == "__main__":
    main()
