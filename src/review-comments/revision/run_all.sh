#!/usr/bin/env bash
# IJIES revision runner (POSIX). Runs from the project root regardless of cwd.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"
exec python "recent-review-comments/revision/code/run_all.py" "$@"
