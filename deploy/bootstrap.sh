#!/usr/bin/env bash
#
# Provision a fresh EC2 instance (Amazon Linux 2023) to run Scuttle Bot.
# Runnable as EC2 user-data at first boot, or by hand after SSHing in as root.
#
# Prerequisites (see deploy/README.md):
#   - the instance has an IAM role with deploy/iam-policy.json attached
#     (grants Secrets Manager + the S3 backup bucket)
#   - the three secrets exist in Secrets Manager:
#       scuttle-bot/discord-token, scuttle-bot/gemini-api-key, scuttle-bot/riot-api-key
# The repo is public, so no git credentials are needed to clone.
set -euo pipefail

APP_DIR=/opt/scuttle-bot
APP_USER=scuttlebot
REPO_URL="${REPO_URL:-https://github.com/bigcoke1/scuttle-bot.git}"

echo "[bootstrap] installing system packages"
dnf -y install git tar gzip

echo "[bootstrap] creating service user ${APP_USER}"
id -u "$APP_USER" &>/dev/null || useradd --system --create-home "$APP_USER"

echo "[bootstrap] cloning ${REPO_URL} into ${APP_DIR}"
if [ ! -d "$APP_DIR/.git" ]; then
    git clone "$REPO_URL" "$APP_DIR"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "[bootstrap] installing rye (manages the pinned Python) for ${APP_USER}"
sudo -u "$APP_USER" bash -c 'test -x "$HOME/.rye/shims/rye" || curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash'
RYE="/home/$APP_USER/.rye/shims/rye"

echo "[bootstrap] installing dependencies (prod only) into .venv"
sudo -u "$APP_USER" bash -c "cd '$APP_DIR' && '$RYE' sync --no-dev"

echo "[bootstrap] restoring sqlite state from S3 (instance role creds)"
# Non-fatal: a first-ever deploy may have nothing to restore; the bot creates a
# fresh db from schema.sql if none is present.
sudo -u "$APP_USER" bash -c "cd '$APP_DIR' && PYTHONPATH=src .venv/bin/python -m scuttle_bot.infra.aws_client restore" || true

echo "[bootstrap] installing systemd units"
cp "$APP_DIR/deploy/scuttle-bot.service" /etc/systemd/system/
cp "$APP_DIR/deploy/scuttle-bot-backup.service" /etc/systemd/system/
cp "$APP_DIR/deploy/scuttle-bot-backup.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now scuttle-bot.service
systemctl enable --now scuttle-bot-backup.timer

echo "[bootstrap] done. Follow logs with: journalctl -u scuttle-bot -f"
