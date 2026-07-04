#!/usr/bin/env bash
# Laptop-side launcher (everything-on-robot topology).
#
# The relay + manager now run ON THE ROBOT (see launch_robot_side.sh), so the
# laptop only records datasets + views the camera. Operator control of the manager
# is done over ssh:  ssh <robot> ; tmux attach -t g1_robot  (keys s/r/f/p/c/x/q).
#
# Default (no args) starts the laptop stack in one tmux window split into tiled
# panes, in startup order (recorder -> viewer):
#
#   ./scripts/launch_laptop_side.sh            # tmux: recorder + viewer, then attach
#   ./scripts/launch_laptop_side.sh all        # same as above
#   ./scripts/launch_laptop_side.sh kill       # tear the tmux session down
#
# Single-component mode (each runs in the foreground of the current terminal; this
# is also what the tmux windows call under the hood):
#
#   ./scripts/launch_laptop_side.sh recorder  # dataset recorder (optional)
#   ./scripts/launch_laptop_side.sh viewer    # camera viewer (optional)
#
# All host flags are baked in from the config block below (override via env). The
# camera, deploy state, AND the manager all live on the robot now, so every source
# host is ROBOT_IP. The recorder/viewer reach them over the ethernet cable
# (192.168.123.x) while the Quest is on the robot's WiFi AP.
#
#   --print / -n   show the exact command(s) without running (safe to test).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Config — override by exporting before you call, e.g. ROBOT_IP=192.168.123.164
# ---------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"       # robot addr (what you ssh to); camera + deploy + manager live here
CAMERA_PORT="${CAMERA_PORT:-5555}"
TASK_PROMPT="${TASK_PROMPT:-pick up the box}"
DATASET_NAME="${DATASET_NAME:-my_session}"
DATA_VENV="${DATA_VENV:-$REPO/.venv_data_collection}"
SESSION="${SESSION:-g1_laptop}"               # tmux session name

# Config vars propagated into each tmux window (so exported overrides survive).
CONFIG_VARS=(REPO ROBOT_IP CAMERA_PORT TASK_PROMPT DATASET_NAME DATA_VENV SESSION)
DEFAULT_COMPONENTS=(recorder viewer)

DRYRUN=0
POS=()
for arg in "$@"; do
    case "$arg" in
        -n|--print) DRYRUN=1 ;;
        *) POS+=("$arg") ;;
    esac
done
MODE="${POS[0]:-}"

# emit <cmd...>: print the shell-quoted command, then run it unless --print.
emit() {
    printf '%q ' "$@"; echo
    [ "$DRYRUN" -eq 1 ] || "$@"
}

# env_prefix: "VAR=val VAR2=val2 ..." for the current config, shell-quoted, so a
# tmux window's fresh shell inherits the same resolved config this process has.
env_prefix() {
    local v esc out=""
    for v in "${CONFIG_VARS[@]}"; do
        printf -v esc '%q' "${!v}"
        out+="$v=$esc "
    done
    printf '%s' "$out"
}

# delay_for <component>: seconds to wait before starting it (enforces order).
delay_for() {
    case "$1" in
        recorder) echo 0 ;;
        viewer)   echo 1 ;;
        *)        echo 0 ;;
    esac
}

# run <venv-or-"-"> <cmd...>: echo the resolved command; run it (foreground)
# unless --print. Activates the venv first when given.
run() {
    local venv="$1"; shift
    echo "# [laptop:$COMPONENT] host flags baked in (ROBOT_IP=$ROBOT_IP)"
    [ "$venv" != "-" ] && echo "source $venv/bin/activate"
    printf '%q ' "$@"; echo   # shell-quoted so --print output is copy-paste-safe
    if [ "$DRYRUN" -eq 1 ]; then return 0; fi
    [ "$venv" != "-" ] && source "$venv/bin/activate"
    exec "$@"
}

# run_single <component>: launch one component in the foreground.
run_single() {
    COMPONENT="$1"
    case "$COMPONENT" in
        recorder)
            # Everything it reads is on the robot now: camera (5555), manager (5556),
            # and deploy state (5557) — all at ROBOT_IP over the ethernet cable.
            run "$DATA_VENV" python "$REPO/gear_sonic/scripts/run_data_exporter.py" \
                --task-prompt "$TASK_PROMPT" \
                --camera-host "$ROBOT_IP" --camera-port "$CAMERA_PORT" \
                --sonic-zmq-host "$ROBOT_IP" --sonic-zmq-port 5556 \
                --state-zmq-host "$ROBOT_IP" --state-zmq-port 5557 \
                --dataset-name "$DATASET_NAME"
            ;;
        viewer)
            run "$DATA_VENV" python "$REPO/gear_sonic/scripts/run_camera_viewer.py" \
                --camera-host "$ROBOT_IP" --camera-port "$CAMERA_PORT"
            ;;
        *)
            usage; exit 2 ;;
    esac
}

# launch_tmux <component...>: all components as tiled panes in one tmux window
# (so you see everything at once), started in order, then attach (or switch-client
# if already inside tmux).
launch_tmux() {
    local comps=("$@")
    command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux not found on this host" >&2; exit 1; }
    if [ "$DRYRUN" -eq 0 ] && tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session '$SESSION' already exists." >&2
        echo "  attach: tmux attach -t $SESSION    kill: $SELF kill" >&2
        exit 1
    fi
    local envp; envp="$(env_prefix)"
    local first=1 c d inner body win_cmd
    for c in "${comps[@]}"; do
        d="$(delay_for "$c")"
        # Run in a subshell so the component's `exec` replaces the subshell only,
        # leaving the pane's outer shell alive to show output after it exits.
        inner="$envp$(printf '%q' "$SELF") $c"
        body="( $inner ); printf '\n[%s exited — press Enter to close] ' $c; read"
        [ "$d" -gt 0 ] && body="echo 'waiting ${d}s for startup order...'; sleep $d; $body"
        # Each pane titles its own border from inside (via \$TMUX_PANE), so the
        # label is always correct regardless of pane creation/layout ordering.
        win_cmd="tmux select-pane -t \"\$TMUX_PANE\" -T $c; $body"
        if [ "$first" -eq 1 ]; then
            emit tmux new-session -d -s "$SESSION" -n stack "$win_cmd"
            first=0
        else
            # Re-tile after each split so panes stay evenly sized and none get
            # too small for the next split.
            emit tmux split-window -t "$SESSION":stack "$win_cmd"
            emit tmux select-layout -t "$SESSION":stack tiled
        fi
    done
    emit tmux set-option -t "$SESSION" pane-border-status top
    emit tmux select-layout -t "$SESSION":stack tiled
    if [ -n "${TMUX:-}" ]; then
        emit tmux switch-client -t "$SESSION"
    else
        emit tmux attach -t "$SESSION"
    fi
}

usage() {
    cat >&2 <<EOF
Usage: $0 [all|recorder|viewer|kill] [--print]

Laptop-side components (everything-on-robot topology):
  (no args)  start recorder + viewer as tiled panes, in order, then attach
  all        same as no args
  recorder   dataset recorder (all sources at ROBOT_IP)  — single, foreground
  viewer     camera viewer                               — single, foreground
  kill       kill the '$SESSION' tmux session

Relay + manager now run on the robot. Drive the manager over ssh:
    ssh <robot> ; tmux attach -t g1_robot     (keys: s r f p c x q)
The Quest connects to the robot's WiFi AP, not the laptop.

Resolved config: ROBOT_IP=$ROBOT_IP  CAMERA_PORT=$CAMERA_PORT
                 DATASET_NAME=$DATASET_NAME  SESSION=$SESSION
Override any config via env vars (see the config block at the top of this file).
EOF
}

case "$MODE" in
    recorder|viewer)
        run_single "$MODE"
        ;;
    kill)
        emit tmux kill-session -t "$SESSION"
        ;;
    ""|all|tmux)
        launch_tmux "${DEFAULT_COMPONENTS[@]}"
        ;;
    *)
        usage; exit 2 ;;
esac
