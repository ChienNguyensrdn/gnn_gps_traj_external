#!/usr/bin/env bash

# Shared, side-effect-free configuration for every TIST2015 runner.
readonly TIST2015_CITIES=(
  Tokyo Nairobi NewYork Sydney CapeTown Paris
  Beijing Mumbai SanFrancisco London SaoPaulo Moscow
)

tist2015_agentmove_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

tist2015_is_city() {
  local requested="$1" city
  for city in "${TIST2015_CITIES[@]}"; do
    [[ "$city" == "$requested" ]] && return 0
  done
  return 1
}

tist2015_require_positive_integer() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer: $value" >&2
    return 2
  fi
}

tist2015_model_slug() {
  local slug="${1//\//-}"
  printf '%s\n' "${slug//:/-}"
}
