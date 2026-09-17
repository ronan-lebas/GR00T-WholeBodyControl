#!/usr/bin/env bash
# Robot-side launcher (everything-on-robot topology).
#
# The whole real-time teleop path runs on the robot: BrainCo hand service, camera
# server, deploy binary, Quest relay, and the Quest manager. Every ZMQ/DDS link is
# localhost on the Jetson. The Quest is CABLED to the robot (see the 'wire'
# component / setup_quest_wire.sh); the laptop only records data + drives the
# manager keyboard over ssh + tmux attach. See ROBOT_TELEOP_README.md.
#
# Default (no args) starts everything at once in one tmux window split into tiled
# panes, one per component, in startup order (hand -> camera -> deploy -> relay ->
# manager). The 'wire' component is NOT started by default — run it once up front
# (re-running it drops the Quest link while the profile is recreated):
#
#   ./scripts/launch_robot_side.sh wire       # bring up the wired Quest link (run once)
#   ./scripts/launch_robot_side.sh            # tmux: hand+camera+deploy+relay+manager
#   ./scripts/launch_robot_side.sh all        # same as above
#   ./scripts/launch_robot_side.sh kill       # tear the tmux session down
#
# Single-component mode (each runs in the foreground of the current terminal; this
# is also what the tmux windows call under the hood):
#
#   ./scripts/launch_robot_side.sh wire       # wired Quest link (setup_quest_wire.sh)
#   ./scripts/launch_robot_side.sh hand       # BrainCo hand service (first)
#   ./scripts/launch_robot_side.sh camera     # composed camera server (ZMQ 5555)
#   ./scripts/launch_robot_side.sh deploy     # g1_deploy_onnx_ref (DDS + ZMQ)
#   ./scripts/launch_robot_side.sh relay      # Quest relay + ego-view image relay
#   ./scripts/launch_robot_side.sh manager    # Quest teleop manager (press s to start)
#
# All host flags are baked in from the config block below (override via env).
#
#   --print / -n   show the exact command(s) without running (safe to test).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Config — override by exporting before you call, e.g. WIRE_IFACE=enx0 EGO_VIEW_CAMERA=oak
# ---------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ZMQ_HOST="${ZMQ_HOST:-localhost}"             # deploy --zmq-host: manager is now on the robot
DEPLOY_TARGET="${DEPLOY_TARGET:-real}"        # deploy.sh arg: 'real' auto-detects the Jetson iface
OUTPUT_TYPE="${OUTPUT_TYPE:-all}"             # 'all' or 'zmq' (zmq skips ROS2)
DEPLOY_EXTRA="${DEPLOY_EXTRA:-}"              # extra deploy.sh flags, e.g. --cp policy/sonic_v1_1/model
EGO_VIEW_CAMERA="${EGO_VIEW_CAMERA:-realsense}"  # oak | realsense | usb | /path/to.mp4
OAK_SERIAL="${OAK_SERIAL:-}"                  # optional --ego-view-device-id
CAMERA_PORT="${CAMERA_PORT:-5555}"
IMAGE_FPS="${IMAGE_FPS:-30}"                  # ego-view image relay cap (relay -> Quest)
ROBOT_IFACE="${ROBOT_IFACE:-}"               # optional -n for the hand service; empty = its own default iface
HAND_USE_SYSTEMD="${HAND_USE_SYSTEMD:-0}"    # 0 = run the binary manually (default); 1 = systemctl restart brainco_hand.service
MANAGER_EXTRA="${MANAGER_EXTRA:-}"           # e.g. "--static-base --log-latency"
WIRE_IFACE="${WIRE_IFACE:-}"                 # USB-ethernet iface to the Quest; empty = auto-detect
WIRE_IP="${WIRE_IP:-192.168.77.1}"           # robot-side gateway; Quest targets WIRE_IP:10000
DATA_VENV="${DATA_VENV:-$REPO/.venv_data_collection}"
TELEOP_VENV="${TELEOP_VENV:-$REPO/.venv_teleop}"
SESSION="${SESSION:-g1_robot}"               # tmux session name

# Config vars propagated into each tmux window (so exported overrides survive).
CONFIG_VARS=(REPO ZMQ_HOST DEPLOY_TARGET OUTPUT_TYPE DEPLOY_EXTRA EGO_VIEW_CAMERA OAK_SERIAL \
             CAMERA_PORT IMAGE_FPS ROBOT_IFACE HAND_USE_SYSTEMD MANAGER_EXTRA \
             WIRE_IFACE WIRE_IP DATA_VENV TELEOP_VENV SESSION)
DEFAULT_COMPONENTS=(hand camera deploy relay manager)

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
        hand)    echo 0 ;;
        camera)  echo 3 ;;
        deploy)  echo 6 ;;
        relay)   echo 9 ;;
        manager) echo 13 ;;
        *)       echo 0 ;;
    esac
}

# run <venv-or-"-"> <workdir-or-"-"> <cmd...>: echo the resolved command; run it
# (foreground) unless --print. Activates the venv and cd's first when given.
run() {
    local venv="$1" workdir="$2"; shift 2
    echo "# [robot:$COMPONENT] host flags baked in (ZMQ_HOST=$ZMQ_HOST, CAMERA_PORT=$CAMERA_PORT)"
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
        wire)
            # One-shot: bring up the direct Quest<->robot ethernet link (DHCP+NAT).
            wire_cmd=("$REPO/gear_sonic_deploy/scripts/setup_quest_wire.sh" --ip "$WIRE_IP")
            [ -n "$WIRE_IFACE" ] && wire_cmd+=(--interface "$WIRE_IFACE")
            run - - "${wire_cmd[@]}"
            ;;
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
            # Manager is now on the robot too, so the command source is localhost.
            deploy_cmd="source scripts/setup_env.sh && ./deploy.sh $DEPLOY_TARGET"
            deploy_cmd="$deploy_cmd --input-type zmq_manager"
            deploy_cmd="$deploy_cmd --zmq-host $ZMQ_HOST --output-type $OUTPUT_TYPE"
            [ -n "$DEPLOY_EXTRA" ] && deploy_cmd="$deploy_cmd $DEPLOY_EXTRA"
            # Deploy runs INSIDE the ROS2 container (docker/run-ros2-dev.sh), which
            # drops into an interactive shell after its banner. --host-net is
            # mandatory here: the deploy binary's DDS must reach the robot's real
            # 192.168.123.x interface, which the bridge default (9fede6d, for
            # rootless docker on the lab machine) cannot carry.
            # In tmux, a watcher waits for the container's ready marker and types
            # the deploy command into the pane; outside tmux, paste it yourself.
            # deploy.sh then stops at its own [Y/n] prompt — the operator types y,
            # which is the last gate before the robot is commanded.
            echo "# [robot:deploy] container: docker/run-ros2-dev.sh --host-net"
            echo "# [robot:deploy] command inside container: $deploy_cmd"
            if [ "$DRYRUN" -eq 1 ]; then
                printf '%q ' "$REPO/gear_sonic_deploy/docker/run-ros2-dev.sh" --host-net; echo
                return 0
            fi
            if [ -n "${TMUX_PANE:-}" ]; then
                (
                    # 'Relays active' is the container startup script's last line,
                    # right before it hands over to the interactive shell.
                    for _ in $(seq 1 1800); do
                        tmux display-message -p -t "$TMUX_PANE" '' >/dev/null 2>&1 || exit 0
                        if tmux capture-pane -p -t "$TMUX_PANE" 2>/dev/null \
                                | grep -q 'Relays active'; then
                            sleep 1
                            tmux send-keys -t "$TMUX_PANE" "$deploy_cmd" C-m
                            exit 0
                        fi
                        sleep 1
                    done
                    echo "[robot:deploy] container never became ready; run manually: $deploy_cmd" >&2
                ) &
            else
                echo "# Not inside tmux — paste the command above into the container shell."
            fi
            exec "$REPO/gear_sonic_deploy/docker/run-ros2-dev.sh" --host-net
            ;;
        relay)
            # Quest relay in Docker with host networking (no NAT hop; 'localhost'
            # reaches the on-board camera server). Quest targets WIRE_IP:10000.
            # No venv — run_quest_relay.py drives Docker via the system python3.
            run - - python3 "$REPO/gear_sonic_deploy/docker/quest_relay/run_quest_relay.py" \
                --network-host \
                --camera-host localhost --camera-port "$CAMERA_PORT" --image-fps "$IMAGE_FPS"
            ;;
        manager)
            # All sources local now: relay (5559) + deploy feedback (5557) on the robot.
            # Attach over ssh (tmux) to drive the keyboard state machine (s/r/f/p/c/x/q).
            # shellcheck disable=SC2086
            run "$TELEOP_VENV" - python "$REPO/gear_sonic/scripts/quest_manager_thread_server.py" \
                --relay-host localhost --relay-port 5559 \
                --port 5556 \
                --feedback-host localhost --feedback-port 5557 $MANAGER_EXTRA
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
Usage: $0 [all|wire|hand|camera|deploy|relay|manager|kill] [--print]

Robot-side components (everything-on-robot topology):
  (no args)  start hand + camera + deploy + relay + manager as tiled panes, then attach
  all        same as no args
  wire       bring up the wired Quest link (setup_quest_wire.sh) — run ONCE, not in defaults
  hand       BrainCo hand service (must be up first)          — single, foreground
  camera     composed camera server, binds ZMQ $CAMERA_PORT           — single, foreground
  deploy     g1_deploy_onnx_ref in the ROS2 container (--zmq-host $ZMQ_HOST)  — single, foreground
  relay      Quest relay (TCP 10000 + ZMQ 5559) + ego-view image relay — single, foreground
  manager    Quest teleop manager (binds 5556; all sources local)      — single, foreground
  kill       kill the '$SESSION' tmux session

Quest is cabled to the robot: plug the adapter in, target $WIRE_IP:10000.
Drive the manager from the laptop with:  ssh <robot> ; tmux attach -t $SESSION
Resolved config: ZMQ_HOST=$ZMQ_HOST  CAMERA_PORT=$CAMERA_PORT  OUTPUT_TYPE=$OUTPUT_TYPE
                 DEPLOY_EXTRA='$DEPLOY_EXTRA'
                 EGO_VIEW_CAMERA=$EGO_VIEW_CAMERA  WIRE_IFACE=${WIRE_IFACE:-<auto>}  WIRE_IP=$WIRE_IP  SESSION=$SESSION
Override any of these via env vars (see the config block at the top of this file).
EOF
}

case "$MODE" in
    wire|hand|camera|deploy|relay|manager)
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
