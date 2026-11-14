"""Build the sandbox image on Modal from MyLocalStack/sandbox/Dockerfile and verify it (HOSTED-3b).

    PYTHONPATH=src uv run python scripts/modal_sandbox_image.py --context <MyLocalStack>/sandbox

The context defaults to $GADS_SANDBOX_IMAGE_CONTEXT. Steps: build (Modal caches layers, so an
unchanged Dockerfile rebuilds in seconds), start one sandbox and time the cold start, then check:
  - pip freeze inside == requirements.lock (+ torch 2.9.1+cpu): no drift, nothing injected
  - scripts/modal_sandbox_check.py: identity/ownership, kaleido, offline embeddings, compiled
    packages, a torch AutoGluon fit, every GADS native preamble
  - the service starts with the image's own command and answers /health
Costs a few cents of sandbox CPU time per run. Exit 0 only if every check passed.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import modal

APP_NAME = "gads-sandbox"
ROOT = Path(__file__).resolve().parent.parent
# Modal runs every exec as root regardless of the Dockerfile's USER; drop privileges explicitly.
SERVICE_USER = ("setpriv", "--reuid=1000", "--regid=1000", "--init-groups", "--")
SERVICE_CMD = (*SERVICE_USER, "sh", "-c",
               "cd /app && HOME=/home/sandboxuser exec uvicorn app:app --host 0.0.0.0 --port 8000")


def sandbox_image(context: Path) -> modal.Image:
    return modal.Image.from_dockerfile(context / "Dockerfile", context_dir=context)


def _run(sb, *cmd, timeout=600):
    p = sb.exec(*cmd, timeout=timeout)
    p.wait()
    return p.returncode, p.stdout.read(), p.stderr.read()


def _lock_set(lock: Path) -> set[str]:
    lines = [l.strip() for l in lock.read_text().splitlines()]
    return {l.lower() for l in lines if l and not l.startswith("#")} | {"torch==2.9.1+cpu"}


def _preambles() -> dict[str, str]:
    sys.path.insert(0, str(ROOT / "src"))
    from gads.knowledge import native

    return {k: v for k, v in vars(native).items() if k.endswith("_PREAMBLE") and isinstance(v, str)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default=os.environ.get("GADS_SANDBOX_IMAGE_CONTEXT"))
    ap.add_argument("--cpu", type=float, default=2.0)
    ap.add_argument("--memory", type=int, default=4096)
    args = ap.parse_args()
    if not args.context:
        ap.error("--context or GADS_SANDBOX_IMAGE_CONTEXT is required")
    context = Path(args.context).resolve()

    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image = sandbox_image(context)
    t0 = time.time()
    with modal.enable_output():
        image.build(app)
    print(f"BUILD {time.time() - t0:.0f}s image={image.object_id}", flush=True)

    t0 = time.time()
    sb = modal.Sandbox.create(app=app, image=image, cpu=args.cpu, memory=args.memory, timeout=1800)
    rc, out, _ = _run(sb, "true")
    print(f"COLD_START {time.time() - t0:.1f}s sandbox={sb.object_id}", flush=True)

    failures = []
    try:
        # 1. package drift
        rc, out, err = _run(sb, "pip", "freeze")
        inside = {l.strip().lower() for l in out.splitlines() if l.strip()}
        want = _lock_set(context / "requirements.lock")
        extra, missing = sorted(inside - want), sorted(want - inside)
        ok = not extra and not missing
        print(f"CHECK freeze_vs_lock {'ok' if ok else 'FAIL'} extra={extra} missing={missing}", flush=True)
        if not ok:
            failures.append("freeze_vs_lock")

        # 2. in-sandbox functional checks
        sb.filesystem.write_text(json.dumps(_preambles()), "/tmp/gads_preambles.json")
        sb.filesystem.write_text((ROOT / "scripts" / "modal_sandbox_check.py").read_text(),
                                 "/tmp/modal_sandbox_check.py")
        # as the kernels will run: uid 1000 (Modal's default exec user is root)
        rc, out, err = _run(sb, *SERVICE_USER, "env", "HOME=/home/sandboxuser", "python",
                            "/tmp/modal_sandbox_check.py", timeout=900)
        print(out, end="")
        if rc != 0:
            print(f"check script rc={rc}: {err[-2000:]}")
            failures.append("check_script")
        failures += [m.group(1) for m in re.finditer(r"^CHECK (\S+) FAIL", out, re.M)]

        # 3. Modal ignores the Dockerfile's USER (exec runs as root), so the service must drop to
        #    sandboxuser itself; otherwise task code could rewrite the root-owned app.py.
        rc, out, _ = _run(sb, *SERVICE_USER, "sh", "-c",
                          "id -u; (echo >> /app/app.py) 2>/dev/null && echo WRITABLE || echo protected")
        ok = out.split() == ["1000", "protected"]
        print(f"CHECK service_user {'ok' if ok else 'FAIL'} {out.split()}", flush=True)
        if not ok:
            failures.append("service_user")

        # 4. the service, started as the backend will start it
        svc = sb.exec(*SERVICE_CMD)
        health = None
        for _ in range(30):
            time.sleep(1)
            rc, out, _ = _run(sb, "python", "-c",
                              "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read().decode())",
                              timeout=10)
            if rc == 0:
                health = out.strip()
                break
        print(f"CHECK service_health {'ok' if health else 'FAIL'} {health}", flush=True)
        if not health:
            failures.append("service_health")
            print(svc.stderr.read()[-2000:] if svc.poll() is not None else "(service still running)")
    finally:
        sb.terminate()

    print("RESULT", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
