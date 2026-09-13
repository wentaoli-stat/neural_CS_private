#!/usr/bin/env bash
# Run the lines of JOBS_FILE as shell commands, at most SLOTS at a time.
# Lines may be appended while the queue runs. Each line's state lives in
# JOBS_FILE.state/<md5>.{cmd,running,done,failed,out}; a line that has a state
# is never started twice. The queue exits once nothing is pending or running,
# unless JOBS_FILE.hold exists (remove it to let the queue drain and exit).
#
#   SLOTS=4 bash scripts/run_job_queue.sh runs/queue/jobs.txt
set -o pipefail

jobs_file="$1"
slots="${SLOTS:-4}"
state="${jobs_file}.state"
mkdir -p "$state"
declare -A pid_of

while true; do
  for key in "${!pid_of[@]}"; do
    if ! kill -0 "${pid_of[$key]}" 2>/dev/null; then
      wait "${pid_of[$key]}"
      rc=$?
      if [[ $rc -eq 0 ]]; then
        mv "$state/$key.running" "$state/$key.done"
      else
        mv "$state/$key.running" "$state/$key.failed"
      fi
      echo "$(date '+%F %T') end rc=$rc $(cat "$state/$key.cmd")" >> "${jobs_file}.log"
      unset "pid_of[$key]"
    fi
  done

  pending=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    key=$(printf '%s' "$line" | md5sum | cut -c1-12)
    if [[ -e "$state/$key.done" || -e "$state/$key.failed" || -e "$state/$key.running" ]]; then
      continue
    fi
    if (( ${#pid_of[@]} < slots )); then
      printf '%s\n' "$line" > "$state/$key.cmd"
      touch "$state/$key.running"
      bash -c "$line" > "$state/$key.out" 2>&1 &
      pid_of[$key]=$!
      echo "$(date '+%F %T') start $line" >> "${jobs_file}.log"
    else
      pending=$((pending + 1))
    fi
  done < "$jobs_file"

  if (( ${#pid_of[@]} == 0 && pending == 0 )) && [[ ! -e "${jobs_file}.hold" ]]; then
    echo "$(date '+%F %T') queue empty; exiting" >> "${jobs_file}.log"
    break
  fi
  sleep 20
done
