#!/usr/bin/env bash
set -euo pipefail

# Create and activate venv with Python 3.12
if [ ! -d "emoji_env" ]; then
  python3.12 -m venv emoji_env
fi
source emoji_env/bin/activate

# Install deps
pip install -r requirements.txt

# macOS camera permissions note
echo "If macOS prompts for camera permissions, approve for Terminal/VSCode." || true

# Run app
python emoji_reactor.py


