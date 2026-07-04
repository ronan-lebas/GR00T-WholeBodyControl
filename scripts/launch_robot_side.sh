#!/usr/bin/env bash
# Robot-side launcher (Option B: deploy on the robot).
#
# Default (no args) starts everything at once in one tmux window split into tiled
# panes, one per component, in the correct startup order (hand -> camera -> deploy):
#
#   ./scripts/launch_robot_side.sh            # tmux: hand + camera + deploy, then attach
#   ./scripts/launch_robot_side.sh all        # same as above
#   ./scripts/launch_robot_side.sh kill       # tear the tmux session down
#
# Single-component mode (each runs in the foreground of the current terminal; this
# is also what the tmux windows call under the hood):
#
#   ./scripts/launch_robot_side.sh hand       # BrainCo hand service (first)
#   ./scripts/launch_robot_side.sh camera     # composed camera server (ZMQ 5555)
#   ./scripts/launch_robot_side.sh deploy     # g1_deploy_onnx_ref (DDS + ZMQ)
#
# All host flags are baked in from the config block below (override via env).
# See deployment_setup.md for the full walkthrough and the networking (NIC) setup.
#
#   --print / -n   show the exact command(s) without running (safe to test).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Config — override by exporting before you call, e.g. LAPTOP_IP=192.168.123.50
# ---------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAPTOP_IP="${LAPTOP_IP:-192.168.123.222}"     # laptop static addr (deploy --zmq-host target)
DEPLOY_TARGET="${DEPLOY_TARGET:-real}"        # deploy.sh arg: 'real' auto-detects the Jetson iface
OUTPUT_TYPE="${OUTPUT_TYPE:-all}"             # 'all' or 'zmq' (zmq skips ROS2)
EGO_VIEW_CAMERA="${EGO_VIEW_CAMERA:-realsense}"  # oak | realsense | usb | /path/to.mp4
OAK_SERIAL="${OAK_SERIAL:-}"                  # optional --ego-view-device-id
CAMERA_PORT="${CAMERA_PORT:-5555}"
ROBOT_IFACE="${ROBOT_IFACE:-}"               # optional -n for the hand service; empty = its own default iface
HAND_USE_SYSTEMD="${HAND_USE_SYSTEMD:-0}"    # 0 = run the binary manually (default); 1 = systemctl restart brainco_hand.service
DATA_VENV="${DATA_VENV:-$REPO/.venv_data_collection}"
SESSION="${SESSION:-g1_robot}"               # tmux session name

# Config vars propagated into each tmux window (so exported overrides survive).
CONFIG_VARS=(REPO LAPTOP_IP DEPLOY_TARGET OUTPUT_TYPE EGO_VIEW_CAMERA OAK_SERIAL \
             CAMERA_PORT ROBOT_IFACE HAND_USE_SYSTEMD DATA_VENV SESSION)
DEFAULT_COMPONENTS=(hand camera deploy)

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
        hand)   echo 0 ;;
        camera) echo 3 ;;
        deploy) echo 6 ;;
        *)      echo 0 ;;
    esac
}

# run <venv-or-"-"> <workdir-or-"-"> <cmd...>: echo the resolved command; run it
# (foreground) unless --print. Activates the venv and cd's first when given.
run() {
    local venv="$1" workdir="$2"; shift 2
    echo "# [robot:$COMPONENT] host flags baked in (LAPTOP_IP=$LAPTOP_IP, CAMERA_PORT=$CAMERA_PORT)"
    [ "$venv" != "-" ] && echo "source $venv/bin/activate"
    [ "$workdir" != "-" ] && echo "cd $workdir"
    printf '%q ' "$@"; echo   # shell-quoted so --print output is copy-paste-safe
    if [ "$DRYRUN" -eq 1 ]; then return 0; fi
    [ "$venv" != "-" ] && source "$venv/bin/activate"
    [ "$workdir" != "-" ] && cd "$workdir"
    exec "$@"
}

# run_single <component>: launch one component in the foreground.
run_single() {
    COMPONENT="$1"
    case "$COMPONENT" in
        hand)
            if [ "$HAND_USE_SYSTEMD" -eq 1 ]; then
                echo "# BrainCo hand service via systemd (HAND_USE_SYSTEMD=1)"
                echo "sudo systemctl restart brainco_hand.service && systemctl status brainco_hand.service"
                [ "$DRYRUN" -eq 1 ] && exit 0
                sudo systemctl restart brainco_hand.service
                exec systemctl status brainco_hand.service
            fi
            # Manual launch (default). No -n unless ROBOT_IFACE is set — the hand
            # service's own default interface works.
            hand_cmd=(sudo ./brainco_hand_server)
            [ -n "$ROBOT_IFACE" ] && hand_cmd+=(-n "$ROBOT_IFACE")
            run - "$REPO/gear_sonic_deploy/thirdparty/brainco_hand_service/bin" "${hand_cmd[@]}"
            ;;
        camera)
            cam_args=(--ego-view-camera "$EGO_VIEW_CAMERA" --port "$CAMERA_PORT")
            [ -n "$OAK_SERIAL" ] && cam_args+=(--ego-view-device-id "$OAK_SERIAL")
            run "$DATA_VENV" "$REPO" python -m gear_sonic.camera.composed_camera "${cam_args[@]}"
            ;;
        deploy)
            run - "$REPO/gear_sonic_deploy" \
                ./deploy.sh "$DEPLOY_TARGET" --zmq-host "$LAPTOP_IP" --output-type "$OUTPUT_TYPE"
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
Usage: $0 [all|hand|camera|deploy|kill] [--print]

Robot-side components (Option B — deploy on the robot):
  (no args)  start hand + camera + deploy in a tmux session, in order, then attach
  all        same as no args
  hand       BrainCo hand service (must be up first)      — single, foreground
  camera     composed camera server, binds ZMQ $CAMERA_PORT       — single, foreground
  deploy     g1_deploy_onnx_ref: DDS + publishes g1_debug to LAPTOP_IP — single, foreground
  kill       kill the '$SESSION' tmux session

Resolved config: LAPTOP_IP=$LAPTOP_IP  CAMERA_PORT=$CAMERA_PORT  OUTPUT_TYPE=$OUTPUT_TYPE
                 EGO_VIEW_CAMERA=$EGO_VIEW_CAMERA  DEPLOY_TARGET=$DEPLOY_TARGET  SESSION=$SESSION
Override any of these via env vars (see the config block at the top of this file).
EOF
}

case "$MODE" in
    hand|camera|deploy)
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
