#!/usr/bin/env bash
# Install the captured reference environment without writing in the live project.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${IMPACTSEG_ENV_NAME:-impactseg-release}"
command -v pyenv >/dev/null || { echo 'Install pyenv and pyenv-virtualenv first.' >&2; exit 2; }
pyenv install -s 3.11.16
if ! pyenv virtualenvs --bare | grep -Fxq "$env_name"; then
  pyenv virtualenv 3.11.16 "$env_name"
fi
PYENV_VERSION="$env_name" pyenv exec python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r "$repo_root/requirements.lock"
PYENV_VERSION="$env_name" pyenv exec python -m pip check
echo "Use: export PYENV_VERSION=$env_name"
