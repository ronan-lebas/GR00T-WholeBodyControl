#!/usr/bin/env python3
"""Run FoundationPose 6D pose estimation on an exported simulation episode.

This automates the FoundationPose docker workflow described in
``third_party/FoundationPose/readme.md``:

  1. Ensure the ``foundationpose`` docker image exists (pull + tag if missing).
  2. Ensure a persistent ``foundationpose`` container is running, with the repo
     mounted so both the FoundationPose code and the exported data are visible.
  3. Build the FoundationPose CUDA/C++ extensions once (``build_all.sh``).
  4. Run ``run_demo.py`` on a recorded episode's ``foundation_pose_data``.

Each setup step is idempotent: a warm run (image present, container up, extensions
already built) skips straight to ``run_demo.py``.

The episode is selected from ``--data-dir`` (a folder under ``outputs/``, e.g.
``outputs/2026-06-12-19-32-55``). If omitted, the most recent folder in
``outputs/`` is used. Inside it, the mesh is ``foundation_pose_data/box.obj`` and
the scene is ``foundation_pose_data/episode_<NNNNNN>/`` (``--episode``, default 0).

Usage:
    python foundation_pose/pose_estimation.py
    python foundation_pose/pose_estimation.py --data-dir outputs/2026-06-12-19-32-55
    python foundation_pose/pose_estimation.py --episode 2 --debug 0
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

# foundation_pose/ lives at the repo root; the repo root is its parent.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FP_DIR = REPO_ROOT / "third_party" / "FoundationPose"
OUTPUTS_DIR = REPO_ROOT / "outputs"

IMAGE = "foundationpose:latest"
UPSTREAM_IMAGE = "wenbowen123/foundationpose"
CONTAINER = "foundationpose"
# Marker written inside the container once extensions are built. Lives in the
# container filesystem (not a mount) so it is correctly tied to *this* container:
# if the container is removed, the in-container build artifacts (e.g. kaolin) are
# gone too, and the marker disappears with them -> we rebuild.
BUILD_MARKER = "/opt/.fp_built"


def run(cmd, **kwargs):
    """Run a command, echoing it; raise on non-zero exit."""
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def capture(cmd):
    """Run a command and return (returncode, stdout.strip())."""
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True
    )
    return proc.returncode, proc.stdout.strip()


# --------------------------------------------------------------------------- #
# Docker setup (idempotent)
# --------------------------------------------------------------------------- #


def image_exists() -> bool:
    rc, out = capture(["docker", "images", "-q", IMAGE])
    return rc == 0 and bool(out)


def ensure_image():
    if image_exists():
        print(f"[setup] image '{IMAGE}' present — skipping pull")
        return
    print(f"[setup] image '{IMAGE}' missing — pulling '{UPSTREAM_IMAGE}'")
    run(["docker", "pull", UPSTREAM_IMAGE])
    run(["docker", "tag", UPSTREAM_IMAGE, IMAGE])


def container_status() -> str | None:
    """Return docker container status ('running', 'exited', ...) or None."""
    rc, out = capture(["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER])
    return out if rc == 0 else None


def ensure_container(mount_dir: Path):
    status = container_status()
    if status == "running":
        print(f"[setup] container '{CONTAINER}' already running — reusing")
        return
    if status == "paused":
        run(["docker", "unpause", CONTAINER])
        return
    if status is not None:  # exited / created
        print(f"[setup] starting existing container '{CONTAINER}' ({status})")
        run(["docker", "start", CONTAINER])
        return

    print(f"[setup] creating container '{CONTAINER}' (mount {mount_dir})")
    # Allow the container to open X windows (run_demo.py uses cv2.imshow when
    # debug>=1). Best-effort; harmless if there is no X server.
    subprocess.run(["xhost", "+"], capture_output=True)
    display = os.environ.get("DISPLAY", "")
    run(
        [
            "docker", "run", "-d",
            "--gpus", "all",
            "--env", "NVIDIA_DISABLE_REQUIRE=1",
            "--network=host",
            "--name", CONTAINER,
            "--cap-add=SYS_PTRACE",
            "--security-opt", "seccomp=unconfined",
            "-v", f"{mount_dir}:{mount_dir}",
            "-v", "/home:/home",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
            "-v", "/tmp:/tmp",
            "--ipc=host",
            "-e", f"DISPLAY={display}",
            IMAGE,
            "sleep", "infinity",
        ]
    )


def extensions_built() -> bool:
    rc, _ = capture(["docker", "exec", CONTAINER, "test", "-f", BUILD_MARKER])
    return rc == 0


def ensure_built(fp_dir: Path):
    if extensions_built():
        print("[setup] extensions already built — skipping build_all.sh")
        return
    print("[setup] building FoundationPose extensions (first run, ~minutes)…")
    run(["docker", "exec", "-w", str(fp_dir), CONTAINER, "bash", "build_all.sh"])
    run(["docker", "exec", CONTAINER, "touch", BUILD_MARKER])


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


def latest_output_dir() -> Path:
    if not OUTPUTS_DIR.is_dir():
        sys.exit(f"[error] outputs directory not found: {OUTPUTS_DIR}")
    subdirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    if not subdirs:
        sys.exit(f"[error] no recording folders found in {OUTPUTS_DIR}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def resolve_paths(args) -> tuple[Path, Path, Path]:
    data_dir = Path(args.data_dir).resolve() if args.data_dir else latest_output_dir()
    if not data_dir.is_dir():
        sys.exit(f"[error] data-dir does not exist: {data_dir}")

    fp_data = data_dir / "foundation_pose_data"
    mesh = fp_data / "box.obj"
    scene = fp_data / f"episode_{args.episode:06d}"

    if not mesh.is_file():
        sys.exit(f"[error] mesh not found: {mesh}")
    if not scene.is_dir():
        sys.exit(f"[error] scene dir not found: {scene}")
    return data_dir, mesh, scene


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Recording folder under outputs/ (default: most recent).",
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index to estimate (default: 0)."
    )
    parser.add_argument(
        "--fp-dir",
        type=str,
        default=str(DEFAULT_FP_DIR),
        help="Path to the FoundationPose checkout (default: third_party/FoundationPose).",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help="run_demo.py debug level (1 = show/overlay; 0 = headless, only ob_in_cam).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI feedback (cv2.imshow windows) by forcing debug level to 0.",
    )
    args = parser.parse_args()

    if args.no_gui:
        args.debug = 0

    fp_dir = Path(args.fp_dir).resolve()
    if not (fp_dir / "run_demo.py").is_file():
        sys.exit(f"[error] run_demo.py not found in --fp-dir: {fp_dir}")

    weights = fp_dir / "weights"
    if not weights.is_dir() or not any(weights.iterdir()):
        raise FileNotFoundError(
            f"FoundationPose network weights missing: {weights}\n"
            "These must live in the submodule. Download them per the 'Data prepare' "
            "section of third_party/FoundationPose/readme.md and place them under "
            "third_party/FoundationPose/weights/."
        )

    data_dir, mesh, scene = resolve_paths(args)
    # Mount a directory that contains both the FoundationPose code and the data,
    # at the same path inside the container (so the absolute paths just work).
    mount_dir = Path(os.path.commonpath([str(fp_dir), str(data_dir)]))

    print(f"[info] FoundationPose dir : {fp_dir}")
    print(f"[info] data dir          : {data_dir}")
    print(f"[info] mesh              : {mesh}")
    print(f"[info] scene             : {scene}")
    print(f"[info] container mount    : {mount_dir}")

    ensure_image()
    ensure_container(mount_dir)
    ensure_built(fp_dir)

    print("[run] launching run_demo.py inside the container…")
    run(
        [
            "docker", "exec", "-w", str(fp_dir), CONTAINER,
            "python", "run_demo.py",
            "--mesh_file", str(mesh),
            "--test_scene_dir", str(scene),
            "--debug", str(args.debug),
        ]
    )
    print(f"\n[done] per-frame poses written to: {scene / 'ob_in_cam'}")


if __name__ == "__main__":
    main()
