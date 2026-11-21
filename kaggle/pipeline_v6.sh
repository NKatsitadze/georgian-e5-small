#!/bin/zsh
# v6 chain: wait for reranker scoring -> v6 training -> download -> benchmark. Stops at the first failure.
#   zsh kaggle/pipeline_v6.sh >> results/pipeline_v6.log 2>&1
set -u
cd "${0:A:h}/.."
say() { echo "$(date +%H:%M) | $*"; }
kg() { perl -e 'alarm 600; exec @ARGV' uv run kaggle "$@"; }
status() { perl -e 'alarm 60; exec @ARGV' uv run kaggle kernels status "nikakatsitadze/$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | tail -1 | sed 's/KernelWorkerStatus\.//'; }
wait_for() {  # wait_for <kernel> <label> <push time>: 0 on COMPLETE, 1 on ERROR/CANCEL; warns if queued > 20 min
  local s start=$(date +%s)
  while true; do
    s=$(status "$1")
    case "$s" in
      COMPLETE) say "$2: COMPLETE"; return 0;;
      ERROR|CANCEL*) say "❌ $2: $s"; return 1;;
      QUEUED) (( $(date +%s) - start > 1200 )) && say "⚠️ $2 queued for over 20 min: resubmit?";;
    esac
    say "$2: ${s:-status unavailable}"
    sleep 300
  done
}

say "▶ pipeline v6 started"
wait_for georgian-teacher-pairs-v6 "reranker scoring v6" || exit 1
mkdir -p data/teacher-pairs-v6
kg kernels output nikakatsitadze/georgian-teacher-pairs-v6 -p data/teacher-pairs-v6 --file-pattern "teacher_pairs_info.json" > /dev/null 2>&1
say "scoring info: $(tr -d '\n ' < data/teacher-pairs-v6/teacher_pairs_info.json)"

kg kernels push -p kaggle/rank_v6 2>&1 | tail -1 | sed "s/^/$(date +%H:%M) | /"
sleep 60
wait_for georgian-rank-v6 "training v6" || exit 1

mkdir -p data/models/rank-v6
kg kernels output nikakatsitadze/georgian-rank-v6 -p data/models/rank-v6 > /dev/null 2>&1
[ -d data/models/rank-v6/student-v6-epoch1 ] || { say "❌ student-v6-epoch1 missing"; exit 1; }
say "v6 info: $(python3 -c "import json; d=json.load(open('data/models/rank-v6/rank_v6_info.json')); print(d['epochs'])")"
say "benchmark v6 started"
uv run python baseline.py --models data/models/rank-v6/student-v6-epoch1 > results/bench_v6.log 2>&1 || { say "❌ benchmark failed"; exit 1; }
say "🏁 benchmark: $(grep -E 'student-v6-epoch1 ' results/baseline.md)"
