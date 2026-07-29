# Deploying Scuttle Bot on AWS (EC2)

Scuttle Bot is a long-running Discord client (it holds a persistent gateway
WebSocket), so it runs best as a single always-on process. This deploys it on a
small EC2 instance under systemd, with secrets in Secrets Manager and state
backed up to S3 — reusing the AWS account/bucket the collection pipeline
already uses (`us-west-1`, account `314722146857`).

Why EC2 and not Fargate/Lambda: Lambda can't hold the gateway connection;
Fargate works but its ephemeral storage would lose the SQLite state without
mounting EFS. A small EC2 instance with an EBS volume is the simplest fit.

## What's in this directory

| File | Purpose |
|---|---|
| `scuttle-bot.service` | systemd unit that runs the bot, restarts on crash |
| `scuttle-bot-backup.service` + `.timer` | daily SQLite backup to S3 |
| `iam-policy.json` | least-privilege policy for the instance role |
| `bootstrap.sh` | one-shot provisioning script (usable as EC2 user-data) |

## One-time AWS setup

1. **Secrets** — all three now exist in Secrets Manager (`us-west-1`):
   `scuttle-bot/riot-api-key`, `scuttle-bot/discord-token`,
   `scuttle-bot/gemini-api-key`. To rotate a value later:
   ```bash
   aws secretsmanager put-secret-value --secret-id scuttle-bot/discord-token --secret-string 'NEW_TOKEN' --region us-west-1
   ```

2. **IAM role** — create a role for EC2 with `iam-policy.json` and an instance
   profile:
   ```bash
   aws iam create-role --role-name scuttle-bot-ec2 \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
   aws iam put-role-policy --role-name scuttle-bot-ec2 \
     --policy-name scuttle-bot --policy-document file://deploy/iam-policy.json
   aws iam create-instance-profile --instance-profile-name scuttle-bot-ec2
   aws iam add-role-to-instance-profile --instance-profile-name scuttle-bot-ec2 --role-name scuttle-bot-ec2
   ```

3. **Instance** — launch a **t4g.small** (ARM, 2 GB) in `us-west-1`, Amazon
   Linux 2023, attach the `scuttle-bot-ec2` instance profile. 2 GB gives
   headroom over the free-tier micro once langchain + scikit-learn + pandas are
   loaded. The instance role means no AWS keys live on the box.

## Provision

The repo is public, so no git credentials are needed. Run `bootstrap.sh` as
root (paste into EC2 user-data at launch, or SSH in and run it). It installs
git + rye, clones the repo, `rye sync`s a prod venv, restores the sqlite state
from S3, and installs + starts the systemd units.

```bash
curl -fsSL https://raw.githubusercontent.com/bigcoke1/scuttle-bot/main/deploy/bootstrap.sh | sudo bash
```

## Verify & operate

```bash
systemctl status scuttle-bot           # is it running?
journalctl -u scuttle-bot -f           # live logs
systemctl list-timers scuttle-bot-backup.timer   # next backup run

# deploy an update
cd /opt/scuttle-bot && sudo -u scuttlebot git pull && sudo -u scuttlebot .venv/bin/rye sync --no-dev
sudo systemctl restart scuttle-bot
```

## Notes

- **Daily reports** fire at 10:30 America/Los_Angeles by default (override with
  `REPORT_TIME` / `REPORT_TIMEZONE` env in `scuttle-bot.service`). This runs via
  a discord.py `tasks.loop`; the old `schedule`-library setup never ticked.
- **State durability**: SQLite on the EBS volume, backed up daily to the
  versioned S3 bucket by the timer. Restore on a fresh box happens
  automatically in `bootstrap.sh`. For zero-maintenance durability you could
  later move to RDS or Litestream — overkill at this scale.
- **Single instance = single point of failure**, which is fine here: discord.py
  auto-reconnects and systemd restarts crashes. Don't run two instances against
  the same bot token.
- **Models**: the served logistic model ships in git, so the bot runs without
  an S3 model restore. The gitignored RF/NN weights are only needed for
  retraining — pull them with `python -m scuttle_bot.ml.model_store restore`.
