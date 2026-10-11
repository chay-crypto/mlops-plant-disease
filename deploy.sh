#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-ubuntu@13.50.239.99}"
KEY="${KEY:-$HOME/.ssh/mlops-key.pem}"

echo "Envoi du code et du modèle vers $HOST..."
rsync -az --delete -e "ssh -i $KEY" \
  --exclude '.venv' --exclude 'data' --exclude 'mlruns' \
  --exclude '.git' --exclude '.env' \
  ./ "$HOST":~/app/

echo "Construction et démarrage sur la VM..."
ssh -i "$KEY" "$HOST" "cd ~/app && docker compose up -d --build && docker compose ps"