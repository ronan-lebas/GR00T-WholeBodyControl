#!/bin/bash
# Bring up a WiFi Access Point (hotspot) ON THE ROBOT so the Quest can connect
# directly to the robot instead of going Quest -> laptop -> robot.
#
# Companion to setup_wifi.sh (which is a WiFi *client* connector). This one turns
# the Jetson's WiFi radio into an AP via NetworkManager (nmcli). The Quest joins
# this AP and targets <AP_IP>:10000 (the quest_relay's ROS-TCP endpoint), so the
# whole teleop path stays on the robot. See deployment_setup.md.
#
# Usage:
#   AP_SSID="g1-teleop" AP_PASS="groot1234" ./setup_ap.sh
#   ./setup_ap.sh --ssid g1-teleop --password groot1234 --band a --ip 192.168.55.1
#   ./setup_ap.sh down          # tear the AP down (restores normal WiFi)
#
# Notes:
#   - A single-radio Jetson cannot be a WiFi client (internet) AND an AP at the
#     same time. During teleop the robot doesn't need WiFi internet — the laptop
#     link is over ethernet (192.168.123.x). If the onboard antenna is weak, use
#     a USB WiFi dongle that supports AP mode and pass its iface via --interface.
#   - Prefer 5 GHz (band a) for lower latency / more bandwidth (matters for the
#     ego-view stream back to the Quest).

set -uo pipefail

# --- Config (env or flags) -------------------------------------------------
AP_SSID="${AP_SSID:-g1-teleop}"
AP_PASS="${AP_PASS:-groot1234}"           # >= 8 chars for WPA2
AP_IFACE="${AP_IFACE:-}"                  # auto-detect wifi iface if empty
AP_CON_NAME="${AP_CON_NAME:-g1_ap}"       # NetworkManager profile name
AP_BAND="${AP_BAND:-a}"                   # a = 5 GHz, bg = 2.4 GHz
AP_CHANNEL="${AP_CHANNEL:-}"              # optional explicit channel
AP_IP="${AP_IP:-192.168.55.1}"            # AP gateway IP (Quest gets DHCP on this /24)

usage() {
    cat << 'USAGE_EOF'
Usage: setup_ap.sh [up|down] [options]

  up (default)   create + activate the AP profile
  down           deactivate the AP profile (restores normal WiFi client use)

Options:
  --ssid <ssid>          AP SSID            (env AP_SSID,    default g1-teleop)
  --password <pw>        WPA2 passphrase    (env AP_PASS,    default groot1234)
  --interface <ifname>   WiFi iface         (env AP_IFACE,   auto-detect)
  --connection <name>    NM profile name    (env AP_CON_NAME,default g1_ap)
  --band <a|bg>          5 GHz (a) / 2.4 (bg) (env AP_BAND,  default a)
  --channel <n>          explicit channel   (env AP_CHANNEL, default auto)
  --ip <ip>              AP gateway IP      (env AP_IP,      default 192.168.55.1)
  -h, --help             Show this help

The Quest then targets  <AP_IP>:10000  (the quest_relay ROS-TCP endpoint).
USAGE_EOF
}

MODE="up"
while [ $# -gt 0 ]; do
    case "$1" in
        up|down) MODE="$1"; shift ;;
        --ssid) AP_SSID="$2"; shift 2 ;;
        --password) AP_PASS="$2"; shift 2 ;;
        --interface) AP_IFACE="$2"; shift 2 ;;
        --connection) AP_CON_NAME="$2"; shift 2 ;;
        --band) AP_BAND="$2"; shift 2 ;;
        --channel) AP_CHANNEL="$2"; shift 2 ;;
        --ip) AP_IP="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[!] Unknown option: $1"; usage; exit 2 ;;
    esac
done

echo "=== [AP Setup Script] Starting (mode: $MODE)... ==="

# --- Teardown --------------------------------------------------------------
if [ "$MODE" = "down" ]; then
    echo "[down] Deactivating AP profile '$AP_CON_NAME'..."
    sudo nmcli connection down "$AP_CON_NAME" 2>/dev/null || true
    echo "[✓] AP down. (Profile kept; delete with: sudo nmcli connection delete $AP_CON_NAME)"
    exit 0
fi

# --- Detect WiFi interface -------------------------------------------------
if [ -z "$AP_IFACE" ]; then
    AP_IFACE=$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="wifi"{print $1; exit}')
fi
if [ -z "$AP_IFACE" ]; then
    echo "[✗] No WiFi interface detected. Provide --interface or AP_IFACE."
    exit 2
fi
echo "[1] Using WiFi interface: $AP_IFACE"

# --- Preflight: does this radio support AP mode? ---------------------------
echo "[2] Checking AP-mode support..."
if command -v iw >/dev/null 2>&1; then
    if iw list 2>/dev/null | grep -A 10 "Supported interface modes" | grep -qw "AP"; then
        echo "[✓] Radio reports AP mode support."
    else
        echo "[✗] This radio does NOT list 'AP' in its supported interface modes."
        echo "    Onboard Jetson WiFi is often client-only — use a USB WiFi dongle that"
        echo "    supports AP mode and pass it via --interface."
        exit 1
    fi
else
    echo "[!] 'iw' not installed — skipping AP-mode check (install with: sudo apt install iw)."
fi

# --- (Re)create the AP profile ---------------------------------------------
echo "[3] Unblocking WiFi and (re)creating AP profile '$AP_CON_NAME'..."
sudo rfkill unblock wifi 2>/dev/null || true
sudo nmcli connection delete "$AP_CON_NAME" 2>/dev/null || true

if ! sudo nmcli connection add type wifi ifname "$AP_IFACE" con-name "$AP_CON_NAME" \
        autoconnect no ssid "$AP_SSID"; then
    echo "[✗] Failed to create AP connection profile."
    exit 1
fi

# AP mode + band + WPA2 + shared IPv4 (NM runs dnsmasq for DHCP on AP_IP/24).
# ipv4.method shared gives DHCP + NAT without stealing the default route, so the
# ethernet link to the laptop (192.168.123.x) stays the primary route.
sudo nmcli connection modify "$AP_CON_NAME" \
    802-11-wireless.mode ap \
    802-11-wireless.band "$AP_BAND" \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$AP_PASS" \
    ipv4.method shared \
    ipv4.addresses "${AP_IP}/24"

[ -n "$AP_CHANNEL" ] && sudo nmcli connection modify "$AP_CON_NAME" 802-11-wireless.channel "$AP_CHANNEL"

echo "[4] Activating AP..."
if ! sudo nmcli connection up "$AP_CON_NAME"; then
    echo "[✗] Failed to bring up the AP. Common causes:"
    echo "    - band/channel not allowed by regulatory domain (try --band bg)"
    echo "    - iface already in use as a client (disconnect it first)"
    exit 1
fi

# --- Report ----------------------------------------------------------------
echo ""
echo "[✓] Access Point is up."
echo "    SSID:      $AP_SSID"
echo "    Password:  $AP_PASS"
echo "    Band:      $AP_BAND    Interface: $AP_IFACE"
echo "    AP IP:     $AP_IP   (Quest gets a DHCP lease on ${AP_IP%.*}.0/24)"
echo ""
echo "    On the Quest: join WiFi '$AP_SSID', then point the Unity app at:"
echo "        $AP_IP:10000"
echo ""
echo "    Tear down with:  $0 down"
echo "=== [AP Setup Script] Completed ==="
