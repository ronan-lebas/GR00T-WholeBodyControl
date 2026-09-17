#!/bin/bash
# Bring up a direct WIRED link between the Quest and the robot, so the whole
# teleop path is Quest --ethernet--> robot.
#
# Usage:
#   ./setup_quest_wire.sh --interface enxXXXXXXXX
#   ./setup_quest_wire.sh                       # auto-detect if unambiguous
#   ./setup_quest_wire.sh down                  # tear the link down
#
# ipv4.method shared makes NetworkManager run dnsmasq (DHCP) + NAT on this iface.
# The Quest's ethernet defaults to DHCP, so nothing is configured on the headset
# beyond pointing the Unity app at <WIRE_IP>:10000. `shared` does not install a
# default route, so the robot's internal 192.168.123.x LAN and the laptop cable
# are unaffected.

set -uo pipefail

# --- Config (env or flags) -------------------------------------------------
WIRE_IFACE="${WIRE_IFACE:-}"                    # USB-ethernet iface the Quest is on
WIRE_CON_NAME="${WIRE_CON_NAME:-quest_wire}"    # NetworkManager profile name
WIRE_IP="${WIRE_IP:-192.168.77.1}"              # robot-side gateway; Quest targets this

# An interface that already carries a real IPv4 address belongs to something else
# (the robot's DDS LAN and the laptop cable are both 192.168.123.x, and an uplink
# NIC may be carrying NFS). The Quest adapter has no address until we give it one,
# so "unaddressed" is the selection rule; 169.254.x is link-local, i.e. still free.
iface_addr() {
    ip -4 -br addr show dev "$1" 2>/dev/null \
        | awk '{for (i = 3; i <= NF; i++) if ($i !~ /^169\.254\./) {print $i; exit}}'
}

usage() {
    cat << 'USAGE_EOF'
Usage: setup_quest_wire.sh [up|down] [options]

  up (default)   create + activate the wired Quest profile (DHCP + NAT via nmcli)
  down           deactivate it

Options:
  --interface <ifname>   USB-ethernet iface  (env WIRE_IFACE, auto-detect)
  --connection <name>    NM profile name     (env WIRE_CON_NAME, default quest_wire)
  --ip <ip>              robot-side gateway  (env WIRE_IP,    default 192.168.77.1)
  -h, --help             Show this help

The Quest then targets  <WIRE_IP>:10000  (the quest_relay ROS-TCP endpoint).
USAGE_EOF
}

MODE="up"
while [ $# -gt 0 ]; do
    case "$1" in
        up|down) MODE="$1"; shift ;;
        --interface) WIRE_IFACE="$2"; shift 2 ;;
        --connection) WIRE_CON_NAME="$2"; shift 2 ;;
        --ip) WIRE_IP="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[!] Unknown option: $1"; usage; exit 2 ;;
    esac
done

echo "=== [Quest Wire Setup] Starting (mode: $MODE)... ==="

if [ "$MODE" = "down" ]; then
    echo "[down] Deactivating '$WIRE_CON_NAME'..."
    sudo nmcli connection down "$WIRE_CON_NAME" 2>/dev/null || true
    echo "[✓] Down. (Profile kept; delete with: sudo nmcli connection delete $WIRE_CON_NAME)"
    exit 0
fi

# --- Pick the interface ----------------------------------------------------
if [ -z "$WIRE_IFACE" ]; then
    mapfile -t candidates < <(
        nmcli -t -f DEVICE,TYPE device status \
            | awk -F: '$2=="ethernet"{print $1}' \
            | while read -r dev; do [ -z "$(iface_addr "$dev")" ] && echo "$dev"; done
    )
    if [ "${#candidates[@]}" -eq 1 ]; then
        WIRE_IFACE="${candidates[0]}"
    elif [ "${#candidates[@]}" -eq 0 ]; then
        echo "[✗] No unaddressed ethernet interface found. Plug the Quest's USB-ethernet"
        echo "    adapter into the robot and re-run, or name it with --interface."
        echo "    Current ethernet interfaces:"
        nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="ethernet"{print $1}' \
            | while read -r dev; do echo "      $dev  ($(iface_addr "$dev"))"; done
        exit 2
    else
        echo "[✗] Several unaddressed interfaces — pick one with --interface:"
        printf '      %s\n' "${candidates[@]}"
        exit 2
    fi
fi

# Explicit --interface still gets the check: re-running with our own profile
# already up is fine, anything else is somebody else's link.
existing_addr="$(iface_addr "$WIRE_IFACE")"
if [ -n "$existing_addr" ] && [ "${existing_addr%/*}" != "$WIRE_IP" ]; then
    echo "[✗] Refusing: '$WIRE_IFACE' already carries $existing_addr, so it belongs to"
    echo "    another link (robot DDS LAN, laptop cable, or an uplink). Pick the"
    echo "    USB-ethernet adapter the Quest is plugged into."
    exit 1
fi
echo "[1] Using interface: $WIRE_IFACE"

if ! ip link show "$WIRE_IFACE" 2>/dev/null | grep -q "state UP\|LOWER_UP"; then
    echo "[!] '$WIRE_IFACE' shows no carrier — check the cable to the Quest adapter"
    echo "    and that the adapter is powered (PD passthrough). Continuing anyway."
fi

# --- (Re)create the profile ------------------------------------------------
echo "[2] (Re)creating profile '$WIRE_CON_NAME'..."
sudo nmcli connection delete "$WIRE_CON_NAME" 2>/dev/null || true

if ! sudo nmcli connection add type ethernet ifname "$WIRE_IFACE" con-name "$WIRE_CON_NAME" \
        autoconnect no \
        ipv4.method shared ipv4.addresses "${WIRE_IP}/24" \
        ipv4.never-default yes ipv6.method disabled; then
    echo "[✗] Failed to create the connection profile."
    exit 1
fi

echo "[3] Activating..."
if ! sudo nmcli connection up "$WIRE_CON_NAME"; then
    echo "[✗] Failed to bring the link up. Check that the adapter is present"
    echo "    (ip -br link) and not claimed by another NM profile."
    exit 1
fi

echo ""
echo "[✓] Wired Quest link is up."
echo "    Interface: $WIRE_IFACE"
echo "    Robot IP:  $WIRE_IP   (Quest gets a DHCP lease on ${WIRE_IP%.*}.0/24)"
echo ""
echo "    On the Quest: plug the ethernet adapter in (no headset network config"
echo "    needed), then point the Unity app at:"
echo "        $WIRE_IP:10000"
echo ""
echo "    Confirm the Quest picked up a lease:  ip neigh show dev $WIRE_IFACE"
echo "    Tear down with:  $0 down"
echo "=== [Quest Wire Setup] Completed ==="
