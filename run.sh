#!/usr/bin/env bash
# Launch the Face Recognition Attendance app.
#
# 1. Sets LD_LIBRARY_PATH so the conda-installed dlib loads its BLAS libs.
# 2. Generates a self-signed HTTPS certificate (camera access requires a
#    secure context: HTTPS or localhost). Skip with:  TLS=0 ./run.sh
# 3. Starts a local TURN server (coturn in docker) so WebRTC works when the
#    phone is on a different network than the server.
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/tmp:/home/bumblebee/miniconda3/pkgs/libcblas-3.11.0-11_h0358290_openblas/lib:/home/bumblebee/miniconda3/lib:${LD_LIBRARY_PATH:-}"

# ---- HTTPS certificate ------------------------------------------------------ #
TLS_ARGS=()
if [ "${TLS:-1}" = "1" ] && command -v openssl >/dev/null 2>&1; then
  mkdir -p .certs
  if [ ! -f .certs/cert.pem ]; then
    san="DNS:localhost,DNS:$(hostname),IP:127.0.0.1"
    for ip in $(hostname -I 2>/dev/null || true); do
      san="$san,IP:$ip"
    done
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout .certs/key.pem -out .certs/cert.pem \
      -subj "/CN=face-attendance" \
      -addext "subjectAltName=$san" >/dev/null 2>&1
    echo "[run.sh] Self-signed HTTPS certificate created in .certs/"
  fi
  TLS_ARGS=(--server.sslCertFile=.certs/cert.pem --server.sslKeyFile=.certs/key.pem)
fi

# ---- Local TURN server (for phones on a different network) ------------------ #
if [ "${TURN:-1}" = "1" ] && command -v docker >/dev/null 2>&1; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'face-attendance-turn'; then
    docker rm -f face-attendance-turn >/dev/null 2>&1 || true
    if ! docker image inspect coturn/coturn:latest >/dev/null 2>&1; then
      docker pull coturn/coturn:latest >/dev/null 2>&1 || true
    fi
    docker run -d --name face-attendance-turn --network=host \
      coturn/coturn:latest \
      --listening-port=3478 --min-port=49152 --max-port=65535 \
      --fingerprint --lt-cred-mech \
      --user=attendance:attendance-secret \
      --no-cli --no-tls >/dev/null 2>&1 \
      && echo "[run.sh] TURN server (coturn) started on port 3478" \
      || echo "[run.sh] WARNING: could not start TURN server (camera still works on the same network)"
  fi
fi

exec streamlit run app.py --server.headless true "${TLS_ARGS[@]}" "$@"
