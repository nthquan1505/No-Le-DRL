#!/usr/bin/env bash
# ==============================================================================
# update.sh - Automated Git Pull & Service Restart for 24/7 Ngrok VM
# ==============================================================================
set -euo pipefail

# 1. Resolve project directory (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " [$(date '+%Y-%m-%d %H:%M:%S')] Starting VM Auto-Update"
echo " Working directory: $SCRIPT_DIR"
echo "=================================================="

# 2. Stop running server first so SQLite database and feed files are released
echo ">> Stopping any running server to unlock database files..."
if systemctl is-active --quiet rss-bot 2>/dev/null; then
    sudo systemctl stop rss-bot || true
elif systemctl is-active --quiet fb-rss 2>/dev/null; then
    sudo systemctl stop fb-rss || true
fi
SERVER_PID=$(pgrep -f "python.*(main\.py\s+serve|server\.py|uvicorn.*server:app)" || true)
if [ -n "$SERVER_PID" ]; then
    echo ">> Terminating background server process (PID: $SERVER_PID)..."
    kill -15 $SERVER_PID 2>/dev/null || kill -9 $SERVER_PID 2>/dev/null || true
    sleep 1
fi

# 3. Preserve live SQLite databases locally on VM (data is NOT synced via GitHub)
TMP_BACKUP_DIR=$(mktemp -d)
for db_file in webapp.db rss_bot.db; do
    if [ -f "$SCRIPT_DIR/$db_file" ]; then
        cp -p "$SCRIPT_DIR/$db_file" "$TMP_BACKUP_DIR/$db_file"
    fi
done

TARGET_BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo deploy_server)}"
if [ -z "$TARGET_BRANCH" ] || [ "$TARGET_BRANCH" = "HEAD" ]; then
    TARGET_BRANCH="deploy_server"
fi

echo ">> Fetching latest code from origin/$TARGET_BRANCH..."
git fetch origin "$TARGET_BRANCH"
echo ">> Resetting code to origin/$TARGET_BRANCH (preserving VM database files)..."
git reset --hard "origin/$TARGET_BRANCH"

for db_file in webapp.db rss_bot.db; do
    if [ -f "$TMP_BACKUP_DIR/$db_file" ]; then
        cp -p "$TMP_BACKUP_DIR/$db_file" "$SCRIPT_DIR/$db_file"
    fi
done
rm -rf "$TMP_BACKUP_DIR"

# 4. Activate Python virtual environment
VENV_DIR=""
for venv_path in "$SCRIPT_DIR/venv" "$SCRIPT_DIR/.venv" "$HOME/venv" "$HOME/.venv"; do
    if [ -f "$venv_path/bin/activate" ]; then
        VENV_DIR="$venv_path"
        break
    fi
done

if [ -n "$VENV_DIR" ]; then
    echo ">> Activating virtual environment: $VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
else
    echo ">> [WARNING] Virtual environment not found. Using system python3."
fi

# 5. Upgrade Python dependencies if requirements.txt changed
if [ -f "requirements.txt" ]; then
    echo ">> Checking and installing updated dependencies..."
    pip install -q -r requirements.txt
fi

# 6. Ensure Playwright browser binaries are present
if command -v playwright &> /dev/null; then
    echo ">> Ensuring Playwright Chromium is installed..."
    playwright install chromium || true
fi

# 7. Restart the background server / scheduler daemon
echo ">> Restarting service daemon..."
if systemctl is-active --quiet rss-bot 2>/dev/null; then
    echo ">> Restarting systemd service: rss-bot"
    sudo systemctl restart rss-bot
elif systemctl is-active --quiet fb-rss 2>/dev/null; then
    echo ">> Restarting systemd service: fb-rss"
    sudo systemctl restart fb-rss
elif command -v pm2 &> /dev/null && pm2 list | grep -q "server"; then
    echo ">> Reloading PM2 process..."
    pm2 restart all
else
    echo ">> Checking for running server / main.py processes..."
    SERVER_PID=$(pgrep -f "python.*(main\.py\s+serve|server\.py|uvicorn.*server:app)" || true)
    if [ -n "$SERVER_PID" ]; then
        echo ">> Stopping existing server (PID: $SERVER_PID)..."
        kill -15 $SERVER_PID 2>/dev/null || kill -9 $SERVER_PID 2>/dev/null || true
        sleep 2
    fi

    echo ">> Starting new server in background (nohup)..."
    nohup python main.py serve --host 0.0.0.0 --port 8000 >> "$SCRIPT_DIR/server.log" 2>&1 &
    sleep 2
    NEW_PID=$(pgrep -f "python.*(main\.py\s+serve|uvicorn.*server:app)" || true)
    echo ">> Server started with PID: $NEW_PID"
fi

echo "=================================================="
echo " [$(date '+%Y-%m-%d %H:%M:%S')] VM Update Completed Successfully"
echo "=================================================="
