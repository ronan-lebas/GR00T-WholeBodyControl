#!/usr/bin/env bash
# Laptop-side launcher (Option B: deploy on the robot).
#
# Default (no args) starts the whole laptop stack at once in one tmux window split
# into tiled panes, one per component, in the correct startup order
# (relay -> manager -> recorder -> viewer):
#
#   ./scripts/launch_laptop_side.sh            # tmux: relay + manager + recorder + viewer, then attach
#   ./scripts/launch_laptop_side.sh all        # same as above
#   ./scripts/launch_laptop_side.sh kill       # tear the tmux session down
#
# Single-component mode (each runs in the foreground of the current terminal; this
# is also what the tmux windows call under the hood):
#
#   ./scripts/launch_laptop_side.sh relay     # Quest relay + ego-view image relay
#   ./scripts/launch_laptop_side.sh manager   # Quest teleop manager (press s to start)
#   ./scripts/launch_laptop_side.sh recorder  # dataset recorder (optional)
#   ./scripts/launch_laptop_side.sh viewer    # camera viewer (optional)
#
# All host flags are baked in from the config block below (override via env).
# The relay stays on the laptop (the Quest reaches the laptop's Wi-Fi IP; it
# cannot reach the robot's 192.168.123.x subnet with the current layout) and its
# ego-view image relay subscribes to the robot's camera at ROBOT_IP.
#
#   --print / -n   show the exact command(s) without running (safe to test).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Config — override by exporting before you call, e.g. ROBOT_IP=192.168.123.164
# ---------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"       # robot addr (what you ssh to); deploy feedback + camera live here
CAMERA_PORT="${CAMERA_PORT:-5555}"
IMAGE_FPS="${IMAGE_FPS:-30}"                   # ego-view image relay cap
TASK_PROMPT="${TASK_PROMPT:-pick up the box}"
DATASET_NAME="${DATASET_NAME:-my_session}"
MANAGER_EXTRA="${MANAGER_EXTRA:-}"            # e.g. "--static-base --log-latency"
TELEOP_VENV="${TELEOP_VENV:-$REPO/.venv_teleop}"
DATA_VENV="${DATA_VENV:-$REPO/.venv_data_collection}"
SESSION="${SESSION:-g1_laptop}"               # tmux session name

# Config vars propagated into each tmux window (so exported overrides survive).
CONFIG_VARS=(REPO ROBOT_IP CAMERA_PORT IMAGE_FPS TASK_PROMPT DATASET_NAME \
             MANAGER_EXTRA TELEOP_VENV DATA_VENV SESSION)
DEFAULT_COMPONENTS=(relay manager recorder viewer)

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
        relay)    echo 0 ;;
        manager)  echo 4 ;;
        recorder) echo 8 ;;
        viewer)   echo 9 ;;
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
        relay)
            # No venv here — run_quest_relay.py drives Docker; use the system python3.
            # --camera-host enables the robot ego-view -> Quest image relay (WS6).
            run - python3 "$REPO/gear_sonic_deploy/docker/quest_relay/run_quest_relay.py" \
                --camera-host "$ROBOT_IP" --camera-port "$CAMERA_PORT" --image-fps "$IMAGE_FPS"
            ;;
        manager)
            # Relay is local; deploy feedback (g1_debug 5557) is on the robot.
            # shellcheck disable=SC2086
            run "$TELEOP_VENV" python "$REPO/gear_sonic/scripts/quest_manager_thread_server.py" \
                --relay-host localhost --relay-port 5559 \
                --port 5556 \
                --feedback-host "$ROBOT_IP" --feedback-port 5557 $MANAGER_EXTRA
            ;;
        recorder)
            # Camera + deploy state on the robot; manager (5556) is local.
            run "$DATA_VENV" python "$REPO/gear_sonic/scripts/run_data_exporter.py" \
                --task-prompt "$TASK_PROMPT" \
                --camera-host "$ROBOT_IP" --camera-port "$CAMERA_PORT" \
                --sonic-zmq-host localhost --sonic-zmq-port 5556 \
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
Usage: $0 [all|relay|manager|recorder|viewer|kill] [--print]

Laptop-side components (Option B — deploy on the robot):
  (no args)  start relay + manager + recorder + viewer as tiled panes, in order, then attach
  all        same as no args
  relay      Quest relay (TCP 10000 + ZMQ 5559) + ego-view image relay — single, foreground
  manager    Quest teleop manager (binds 5556; feedback from ROBOT_IP)  — single, foreground
  recorder   dataset recorder                                           — single, foreground
  viewer     camera viewer                                              — single, foreground
  kill       kill the '$SESSION' tmux session

Resolved config: ROBOT_IP=$ROBOT_IP  CAMERA_PORT=$CAMERA_PORT  IMAGE_FPS=$IMAGE_FPS
                 DATASET_NAME=$DATASET_NAME  MANAGER_EXTRA='$MANAGER_EXTRA'  SESSION=$SESSION
Point the Quest Unity app at THIS LAPTOP's Wi-Fi/LAN IP : 10000.
Override any config via env vars (see the config block at the top of this file).
EOF
}

case "$MODE" in
    relay|manager|recorder|viewer)
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
