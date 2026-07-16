#!/bin/bash
# Snapshot pending Slurm jobs and generate a replay script to resubmit them.
# Optional: cancel those pending jobs after snapshot.
#
# Usage examples:
#   ./slurm_snapshot_requeue_pending.sh
#   ./slurm_snapshot_requeue_pending.sh --cancel
#   ./slurm_snapshot_requeue_pending.sh --partition tier3 --cancel
#   ./slurm_snapshot_requeue_pending.sh --snapshot-dir /path/to/snap

set -Eeuo pipefail

TARGET_USER="${USER:-}"
PARTITION=""
SNAPSHOT_DIR=""
DO_CANCEL=0

usage() {
  cat <<'EOF'
Usage: slurm_snapshot_requeue_pending.sh [options]

Options:
  --user <user>           Slurm user to snapshot (default: $USER)
  --partition <name>      Only include pending jobs in this partition
  --snapshot-dir <path>   Output directory (default: ./slurm_pending_snapshot_<timestamp>)
  --cancel                Cancel pending jobs after successful snapshot
  -h, --help              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      TARGET_USER="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --snapshot-dir)
      SNAPSHOT_DIR="$2"
      shift 2
      ;;
    --cancel)
      DO_CANCEL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TARGET_USER}" ]]; then
  echo "[ERROR] --user not provided and \$USER is empty." >&2
  exit 2
fi

if [[ -z "${SNAPSHOT_DIR}" ]]; then
  SNAPSHOT_DIR="./slurm_pending_snapshot_$(date +%Y%m%d_%H%M%S)"
fi
SNAPSHOT_DIR="$(mkdir -p "${SNAPSHOT_DIR}" && cd "${SNAPSHOT_DIR}" && pwd)"
JOBS_DIR="${SNAPSHOT_DIR}/jobs"
mkdir -p "${JOBS_DIR}"

SQUEUE_ARGS=(-u "${TARGET_USER}" -h -t PD -o "%i")
if [[ -n "${PARTITION}" ]]; then
  SQUEUE_ARGS+=(-p "${PARTITION}")
fi

mapfile -t JOB_IDS < <(squeue "${SQUEUE_ARGS[@]}" | awk 'NF>0 {print $1}')

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
  echo "[INFO] No pending jobs found for user=${TARGET_USER}${PARTITION:+ partition=${PARTITION}}."
  echo "[INFO] Snapshot dir: ${SNAPSHOT_DIR}"
  exit 0
fi

REQUEUE_SCRIPT="${SNAPSHOT_DIR}/requeue.sh"
JOB_TABLE="${SNAPSHOT_DIR}/jobs.tsv"
RAW_JOB_INFO="${SNAPSHOT_DIR}/scontrol_show_job.txt"

cat > "${REQUEUE_SCRIPT}" <<'EOF'
#!/bin/bash
set -Eeuo pipefail
EOF

echo -e "job_id\tjob_name\tpartition\tcommand\tworkdir\tbatch_script\trequeue_cmd" > "${JOB_TABLE}"
: > "${RAW_JOB_INFO}"

get_field() {
  local line="$1"
  local key="$2"
  awk -v k="${key}" '{
    for (i=1; i<=NF; i++) {
      split($i, a, "=")
      if (a[1] == k) {
        sub(a[1]"=", "", $i)
        print $i
        exit
      }
    }
  }' <<< "${line}"
}

nullish() {
  local v="${1:-}"
  [[ -z "${v}" || "${v}" == "(null)" || "${v}" == "N/A" || "${v}" == "Unknown" ]]
}

for job_id in "${JOB_IDS[@]}"; do
  line="$(scontrol show job -o "${job_id}" || true)"
  if [[ -z "${line}" ]]; then
    echo "[WARN] Could not read job ${job_id}; skipping."
    continue
  fi
  echo "${line}" >> "${RAW_JOB_INFO}"

  job_name="$(get_field "${line}" "JobName")"
  partition_val="$(get_field "${line}" "Partition")"
  account="$(get_field "${line}" "Account")"
  qos="$(get_field "${line}" "QOS")"
  time_limit="$(get_field "${line}" "TimeLimit")"
  ntasks="$(get_field "${line}" "NumTasks")"
  cpus_per_task="$(get_field "${line}" "CPUs/Task")"
  min_mem_node="$(get_field "${line}" "MinMemoryNode")"
  gres="$(get_field "${line}" "Gres")"
  command="$(get_field "${line}" "Command")"
  workdir="$(get_field "${line}" "WorkDir")"
  stdout_path="$(get_field "${line}" "StdOut")"
  stderr_path="$(get_field "${line}" "StdErr")"

  batch_script="${JOBS_DIR}/job_${job_id}.sbatch"
  if ! scontrol write batch_script "${job_id}" "${batch_script}" >/dev/null 2>&1; then
    # Fallback: if we cannot dump, still keep command path reference.
    batch_script=""
  fi

  cmd=(sbatch)
  nullish "${job_name}"      || cmd+=(--job-name="${job_name}")
  nullish "${partition_val}" || cmd+=(--partition="${partition_val}")
  nullish "${account}"       || cmd+=(--account="${account}")
  nullish "${qos}"           || cmd+=(--qos="${qos}")
  nullish "${time_limit}"    || cmd+=(--time="${time_limit}")
  nullish "${ntasks}"        || cmd+=(--ntasks="${ntasks}")
  nullish "${cpus_per_task}" || cmd+=(--cpus-per-task="${cpus_per_task}")
  nullish "${min_mem_node}"  || cmd+=(--mem="${min_mem_node}")
  nullish "${gres}"          || cmd+=(--gres="${gres}")
  nullish "${stdout_path}"   || cmd+=(--output="${stdout_path}")
  nullish "${stderr_path}"   || cmd+=(--error="${stderr_path}")
  nullish "${workdir}"       || cmd+=(--chdir="${workdir}")

  if [[ -n "${batch_script}" && -f "${batch_script}" ]]; then
    cmd+=("${batch_script}")
  else
    if nullish "${command}"; then
      echo "[WARN] Job ${job_id} has no usable batch script or command; skipping replay entry."
      continue
    fi
    cmd+=("${command}")
  fi

  printf '%q ' "${cmd[@]}" >> "${REQUEUE_SCRIPT}"
  echo >> "${REQUEUE_SCRIPT}"

  requeue_cmd_print="$(printf '%q ' "${cmd[@]}")"
  echo -e "${job_id}\t${job_name}\t${partition_val}\t${command}\t${workdir}\t${batch_script}\t${requeue_cmd_print}" >> "${JOB_TABLE}"
done

chmod +x "${REQUEUE_SCRIPT}"

echo "[INFO] Snapshot complete."
echo "[INFO] Pending jobs captured: ${#JOB_IDS[@]}"
echo "[INFO] Snapshot dir: ${SNAPSHOT_DIR}"
echo "[INFO] Replay script: ${REQUEUE_SCRIPT}"
echo "[INFO] Job table: ${JOB_TABLE}"

if [[ "${DO_CANCEL}" -eq 1 ]]; then
  echo "[INFO] Cancelling pending jobs..."
  scancel "${JOB_IDS[@]}"
  echo "[INFO] Cancelled pending jobs."
fi

echo
echo "To re-submit later:"
echo "  bash '${REQUEUE_SCRIPT}'"

