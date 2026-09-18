#!/usr/bin/env bash
# Set QNN_SDK_ROOT from QNN/<version>/ or an existing environment variable.
# Linux amd64 only. Intended to be sourced:  source scripts/qnn-env.sh

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: QNN host tools are Linux amd64 ELF. This OS/arch is $(uname -s) $(uname -m)." >&2
  return 1 2>/dev/null || exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${QNN_SDK_ROOT:-}" && -n "${QAIRT_SDK_ROOT:-}" ]]; then
  QNN_SDK_ROOT="$QAIRT_SDK_ROOT"
fi
if [[ -z "${QNN_SDK_ROOT:-}" && -n "${DASHADAS_QNN_SDK:-}" ]]; then
  QNN_SDK_ROOT="$DASHADAS_QNN_SDK"
fi
if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  if [[ -n "${DASHADAS_QNN_VERSION:-}" && -f "$ROOT/QNN/${DASHADAS_QNN_VERSION}/QNN_README.txt" ]]; then
    QNN_SDK_ROOT="$ROOT/QNN/${DASHADAS_QNN_VERSION}"
  else
    newest=""
    for dir in "$ROOT"/QNN/*/; do
      [[ -f "${dir}QNN_README.txt" || -f "${dir}bin/envsetup.sh" ]] || continue
      if [[ -z "$newest" || "${dir%/}" > "${newest%/}" ]]; then
        newest="$dir"
      fi
    done
    if [[ -n "$newest" ]]; then
      QNN_SDK_ROOT="$(cd "$newest" && pwd)"
    fi
  fi
fi

if [[ -z "${QNN_SDK_ROOT:-}" || ! -d "$QNN_SDK_ROOT" ]]; then
  echo "error: QNN SDK not found. Copy it to QNN/<version>/ (see QNN/README.md)." >&2
  return 1 2>/dev/null || exit 1
fi

export QNN_SDK_ROOT
export QAIRT_SDK_ROOT="$QNN_SDK_ROOT"
# Qualcomm envsetup.sh concatenates PYTHONPATH/LD_LIBRARY_PATH; with `set -u` they must exist.
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export SNPE_ROOT="${SNPE_ROOT:-}"
# shellcheck disable=SC1091
source "$QNN_SDK_ROOT/bin/envsetup.sh"
echo "Using QNN_SDK_ROOT=$QNN_SDK_ROOT"
