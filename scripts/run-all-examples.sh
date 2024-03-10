#!/bin/bash

# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
set -e -x -v -u -o pipefail

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source "${SCRIPT_DIR}/utilities/common.sh"

export COMFY_API_URL=${COMFY_API_URL:-""}

if [[ -z "${COMFY_API_URL}" ]]; then
  echo -e "${RED}COMFY_API_URL is not set${NC}"
  # trunk-ignore(shellcheck/SC2128)
  # trunk-ignore(shellcheck/SC2209)
  [[ $0 == "${BASH_SOURCE}" ]] && EXIT=exit || EXIT=return
  ${EXIT} 1
fi

export COMFY_INSTALL_FILE_URL=${COMFY_INSTALL_FILE_URL:-""}

VENV_PATH="${PWD}/.venv" source "${PROJ_PATH}/scripts/utilities/ensure-venv.sh"
REQS="${PWD}/requirements.txt" source "${PROJ_PATH}/scripts/utilities/ensure-reqs.sh"

export PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH=${PYTHONPATH}:${PWD}


INPUT_DIR="${PWD}/.deleteme/.data/input"
OUTPUT_DIR="${PWD}/.deleteme/.data/output"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${INPUT_DIR}"

ARGS=(
  "--workflow" "comfylowda/assets/sdxlturbo_example.json"
  "--api-workflow" "comfylowda/assets/sdxlturbo_example_api.json"
  "--object-info" "comfylowda/assets/object_info.json"
  "-fs" "{\"pfx\":\"file://${OUTPUT_DIR}/\",\"proto\":\"file\",\"mode\":\"w\"}"
  "-fs" "{\"pfx\":\"file://${INPUT_DIR}/\",\"proto\":\"file\",\"mode\":\"r\"}"
  "-om" '{"name":"Preview Image","node":25,"field":"images[0]","pp":"file","spec":{"mode":"TRIPLET","pfx":null},"user_json_spec":"ANY","user_value":null}'
  "-i" "{\"name\":\"Preview Image\",\"value\":\"file://${OUTPUT_DIR}/output.png\"}"
)

python -m comfylowda.cli "${ARGS[@]}"

# if tiv is a valid command
if command -v tiv &> /dev/null; then
  tiv -w 80 -h 80 "${OUTPUT_DIR}/output.png"
fi

echo -e "${GREEN}All examples ran successfully${NC}"
