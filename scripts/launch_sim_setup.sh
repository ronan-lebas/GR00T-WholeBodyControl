#!/usr/bin/env bash
# Sim manipulation setup launcher (no robot: everything runs on this machine).
#
# Deploys the full teleop stack as if on the robot, but against MuJoCo. The sim
# spawns a static table + graspable box in front of the (static-base) robot and
# publishes the ego view on ZMQ 5555 exactly like the robot's camera server.
#
# Default (no args) starts everything at once in one tmux window split into tiled
# panes, one per component, in the correct startup order
# (sim -> deploy -> relay -> manager -> recorder -> viewer):
#
#   ./scripts/launch_sim_setup.sh            # tmux: full sim stack, then attach
#   ./scripts/launch_sim_setup.sh all        # same as above
#   ./scripts/launch_sim_setup.sh --mock-quest  # full stack, but mock streamer instead of relay+manager
#   ./scripts/launch_sim_setup.sh kill       # tear the tmux session down
#
# Single-component mode (each runs in the foreground of the current terminal; this
# is also what the tmux panes call under the hood):
#
#   ./scripts/launch_sim_setup.sh sim        # MuJoCo sim + table/box + camera pub (ZMQ 5555)
#   ./scripts/launch_sim_setup.sh deploy     # docker/run-ros2-dev.sh container; deploy runs inside
#   ./scripts/launch_sim_setup.sh relay      # Quest relay + ego-view image relay (headset)
#   ./scripts/launch_sim_setup.sh manager    # Quest teleop manager (press s to start)
#   ./scripts/launch_sim_setup.sh recorder   # dataset recorder (optional)
#   ./scripts/launch_sim_setup.sh viewer     # camera viewer (optional)
#   ./scripts/launch_sim_setup.sh mock       # mock_quest_streamer INSTEAD of relay+manager
#
# Scene resets during teleop (from the manager pane): 'b' puts the box back on the
# table (between episodes); '0' full sim reset (recovery — pause with 'p' first).
#
# All host flags are baked in from the config block below (override via env).
#
#   --print / -n   show the exact command(s) without running (safe to test).
#   --mock-quest   full-stack modes: run mock_quest_streamer in place of relay+manager
#                  (no headset needed — the mock drives canned teleop on 5556).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Config — override by exporting before you call, e.g. RENDER_DEPTH_SEG=1
# ---------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CAMERA_PORT="${CAMERA_PORT:-5555}"
IMAGE_FPS="${IMAGE_FPS:-30}"                   # ego-view image relay cap (Quest passthrough)
TASK_PROMPT="${TASK_PROMPT:-pick up the box}"
RENDER_DEPTH_SEG="${RENDER_DEPTH_SEG:-0}"     # 1 = publish ego depth + box seg (FoundationPose)
SIM_EXTRA="${SIM_EXTRA:-}"                     # extra run_sim_loop.py flags, e.g. "--table-height 0.8"
# Deploy runs INSIDE the docker container: localhost there is the container, so the
# manager (5556, bound on this host) is reached via host.docker.internal.
DEPLOY_ZMQ_HOST="${DEPLOY_ZMQ_HOST:-host.docker.internal}"
DEPLOY_EXTRA="${DEPLOY_EXTRA:-}"              # extra deploy.sh flags (inside the container)
MANAGER_EXTRA="${MANAGER_EXTRA:---static-base}"  # default: static base (robot doesn't walk)
SIM_VENV="${SIM_VENV:-$REPO/.venv_sim}"
TELEOP_VENV="${TELEOP_VENV:-$REPO/.venv_teleop}"
DATA_VENV="${DATA_VENV:-$REPO/.venv_data_collection}"
SESSION="${SESSION:-g1_sim}"                   # tmux session name

# Config vars propagated into each tmux window (so exported overrides survive).
CONFIG_VARS=(REPO CAMERA_PORT IMAGE_FPS TASK_PROMPT RENDER_DEPTH_SEG \
             SIM_EXTRA DEPLOY_ZMQ_HOST DEPLOY_EXTRA MANAGER_EXTRA SIM_VENV TELEOP_VENV \
             DATA_VENV SESSION)
DEFAULT_COMPONENTS=(sim deploy relay manager recorder viewer)

DRYRUN=0
MOCK_QUEST=0   # --mock-quest: swap relay+manager for the mock_quest_streamer
POS=()
for arg in "$@"; do
    case "$arg" in
        -n|--print) DRYRUN=1 ;;
        --mock-quest) MOCK_QUEST=1 ;;
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
        deploy)   echo 0 ;;
        sim)      echo 5 ;;
        relay)    echo 7 ;;
        manager)  echo 10 ;;
        mock)     echo 10 ;;
        recorder) echo 14 ;;
        viewer)   echo 15 ;;
        *)        echo 0 ;;
    esac
}

# run <venv-or-"-"> <workdir-or-"-"> <cmd...>: echo the resolved command; run it
# (foreground) unless --print. Activates the venv and cd's first when given.
run() {
    local venv="$1" workdir="$2"; shift 2
    echo "# [sim:$COMPONENT] all hosts are localhost (CAMERA_PORT=$CAMERA_PORT)"
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
        sim)
            # --detach-gantry: start with the robot released from the virtual gantry
            # (same as pressing '9' in the viewer), so it stands on the policy alone.
            sim_args=(--table --box --detach-gantry --enable-image-publish --enable-offscreen
                      --camera-port "$CAMERA_PORT")
            [ "$RENDER_DEPTH_SEG" -eq 1 ] && sim_args+=(--render-depth-seg)
            # shellcheck disable=SC2086
            run "$SIM_VENV" "$REPO" python gear_sonic/scripts/run_sim_loop.py \
                "${sim_args[@]}" $SIM_EXTRA
            ;;
        deploy)
            # Deploy runs INSIDE the ROS2 docker container (docker/run-ros2-dev.sh,
            # which drops into an interactive shell after its setup banner). When in
            # a tmux pane, a background watcher waits for the container's ready
            # marker AND for the sim to be up, then types the deploy command into the
            # pane; outside tmux, run the printed command manually once both are up.
            #
            # Order matters on the lab machine: the deploy container must come up
            # first, then the sim, and only then './deploy.sh' — the deploy binary
            # connects to the sim's SDK/ZMQ channels, so they must already exist.
            # The container starts first via delay_for (deploy=0 < sim=5); the sim
            # gate below holds the './deploy.sh' auto-type until the sim has bound
            # its ego-view port (CAMERA_PORT), guaranteeing container -> sim -> deploy.
            # shellcheck disable=SC2086
            # --input-type zmq_manager: read teleop from the quest manager's ZMQ
            # publisher (this is what makes 's' in the manager pane feed the policy;
            # deploy.sh's default 'manager' input type leaves the poller with no data).
            # --yes: skip deploy.sh's interactive confirmation (sim-only, no robot).
            deploy_cmd="source scripts/setup_env.sh && ./deploy.sh sim --input-type zmq_manager --zmq-host $DEPLOY_ZMQ_HOST --yes"
            [ -n "$DEPLOY_EXTRA" ] && deploy_cmd="$deploy_cmd $DEPLOY_EXTRA"
            echo "# [sim:deploy] container: docker/run-ros2-dev.sh"
            echo "# [sim:deploy] command inside container: $deploy_cmd"
            if [ "$DRYRUN" -eq 1 ]; then
                printf '%q ' "$REPO/gear_sonic_deploy/docker/run-ros2-dev.sh"; echo
                return 0
            fi
            if [ -n "${TMUX_PANE:-}" ]; then
                (
                    # 'Relays active' is the last line of the container's startup
                    # script, right before it hands over to the interactive shell.
                    for _ in $(seq 1 1800); do
                        tmux display-message -p -t "$TMUX_PANE" '' >/dev/null 2>&1 || exit 0
                        if tmux capture-pane -p -t "$TMUX_PANE" 2>/dev/null \
                                | grep -q 'Relays active'; then
                            # Container is up. Hold until the sim has bound its
                            # ego-view port so the deploy script starts last. Bounded
                            # wait so a standalone 'deploy' (no sim) still proceeds.
                            for _ in $(seq 1 600); do
                                if (exec 3<>"/dev/tcp/127.0.0.1/$CAMERA_PORT") 2>/dev/null; then
                                    break
                                fi
                                sleep 1
                            done
                            sleep 1
                            tmux send-keys -t "$TMUX_PANE" "$deploy_cmd" C-m
                            exit 0
                        fi
                        sleep 1
                    done
                    echo "[sim:deploy] container never became ready; run manually: $deploy_cmd" >&2
                ) &
            else
                echo "# Not inside tmux — paste the command above into the container shell."
            fi
            exec "$REPO/gear_sonic_deploy/docker/run-ros2-dev.sh"
            ;;
        relay)
            # No venv here — run_quest_relay.py drives Docker; use the system python3.
            # --camera-host localhost: the sim publishes the ego view on this machine.
            run - - python3 "$REPO/gear_sonic_deploy/docker/quest_relay/run_quest_relay.py" \
                --camera-host localhost --camera-port "$CAMERA_PORT" --image-fps "$IMAGE_FPS"
            ;;
        manager)
            # 'b' = box reset, '0' = full sim reset (shipped via manager_state to the sim).
            # shellcheck disable=SC2086
            run "$TELEOP_VENV" - python "$REPO/gear_sonic/scripts/quest_manager_thread_server.py" \
                --relay-host localhost --relay-port 5559 \
                --port 5556 \
                --feedback-host localhost --feedback-port 5557 $MANAGER_EXTRA
            ;;
        recorder)
            run "$DATA_VENV" - python "$REPO/gear_sonic/scripts/run_data_exporter.py" \
                --task-prompt "$TASK_PROMPT" \
                --camera-host localhost --camera-port "$CAMERA_PORT" \
                --sonic-zmq-host localhost --sonic-zmq-port 5556 \
                --state-zmq-host localhost --state-zmq-port 5557 \
                --hand-type brainco
            ;;
        viewer)
            run "$DATA_VENV" - python "$REPO/gear_sonic/scripts/run_camera_viewer.py" \
                --camera-host localhost --camera-port "$CAMERA_PORT"
            ;;
        mock)
            # Binds 5556 itself — use INSTEAD of relay+manager, never alongside them.
            # The mock starts streaming teleop the instant it launches, so it must
            # come up only AFTER the policy (deploy) has finished initializing.
            # The 10s startup delay isn't enough (policy init time varies), so gate
            # on an explicit go-ahead instead of a timer.
            if [ "$DRYRUN" -eq 0 ]; then
                read -r -p "[sim:mock] Wait until the policy has finished initializing, then press Enter to start the mock streamer... "
            fi
            run "$TELEOP_VENV" "$REPO" python gear_sonic/scripts/mock_quest_streamer.py \
                --hand brainco
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
Usage: $0 [all|sim|deploy|relay|manager|recorder|viewer|mock|kill] [--mock-quest] [--print]

Sim manipulation stack (no robot — everything on this machine):
  (no args)  start sim + deploy + relay + manager + recorder + viewer as tiled panes
  all        same as no args
  --mock-quest  (with all/no-args) swap relay+manager for mock_quest_streamer
  sim        MuJoCo sim: table + box + ego-view publisher on ZMQ $CAMERA_PORT — single, foreground
  deploy     docker/run-ros2-dev.sh container; inside, sources setup_env.sh and runs
             './deploy.sh sim --input-type zmq_manager --zmq-host $DEPLOY_ZMQ_HOST --yes' (auto-typed in tmux)
  relay      Quest relay (TCP 10000 + ZMQ 5559) + ego-view image relay    — single, foreground
  manager    Quest teleop manager (binds 5556; b=box reset, 0=full reset) — single, foreground
  recorder   dataset recorder                                             — single, foreground
  viewer     camera viewer                                                — single, foreground
  mock       mock_quest_streamer (binds 5556) — run INSTEAD of relay+manager
  kill       kill the '$SESSION' tmux session

Resolved config: CAMERA_PORT=$CAMERA_PORT  RENDER_DEPTH_SEG=$RENDER_DEPTH_SEG
                 MANAGER_EXTRA='$MANAGER_EXTRA'  SIM_EXTRA='$SIM_EXTRA'  SESSION=$SESSION
Point the Quest Unity app at THIS machine's Wi-Fi/LAN IP : 10000.
Override any config via env vars (see the config block at the top of this file).
EOF
}

case "$MODE" in
    sim|deploy|relay|manager|recorder|viewer|mock)
        run_single "$MODE"
        ;;
    kill)
        # The relay is started attached under its tmux pane, but the docker daemon
        # keeps its container alive after the pane (and run_quest_relay.py, which
        # doesn't handle SIGHUP) are killed — so tearing down tmux leaks it and its
        # ports (10000/5559) stay bound. Force-remove it too. Both steps are
        # best-effort so cleanup runs even if one target is already gone.
        if [ "$DRYRUN" -eq 1 ]; then
            printf '%q ' tmux kill-session -t "$SESSION"; echo
            printf '%q ' docker rm -f quest-relay; echo
        else
            tmux kill-session -t "$SESSION" 2>/dev/null || true
            docker rm -f quest-relay 2>/dev/null || true
            echo "killed tmux session '$SESSION'; removed quest-relay container (if present)"
        fi
        ;;
    ""|all|tmux)
        components=("${DEFAULT_COMPONENTS[@]}")
        if [ "$MOCK_QUEST" -eq 1 ]; then
            # mock_quest_streamer binds 5556 itself, standing in for the whole
            # relay+manager teleop chain — drop both and start it in their place.
            components=()
            for c in "${DEFAULT_COMPONENTS[@]}"; do
                case "$c" in
                    relay)   components+=(mock) ;;   # take relay's slot
                    manager) ;;                      # drop (mock covers it)
                    *)       components+=("$c") ;;
                esac
            done
        fi
        launch_tmux "${components[@]}"
        ;;
    *)
        usage; exit 2 ;;
esac
