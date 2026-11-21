#!/bin/zsh
# Overnight v5 chain: wait for the translation job -> build pairs -> upload -> teacher scoring ->
# v5 training -> download -> benchmark. Stops at the first failure. Run from the project root:
#   zsh kaggle/pipeline_v5.sh >> results/pipeline_v5.log 2>&1
set -u
cd "${0:A:h}/.."
say() { echo "$(date +%H:%M) | $*"; }
kg() { perl -e 'alarm 600; exec @ARGV' uv run kaggle "$@"; }
status() { perl -e 'alarm 60; exec @ARGV' uv run kaggle kernels status "nikakatsitadze/$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | tail -1 | sed 's/KernelWorkerStatus\.//'; }
wait_for() {  # wait_for <kernel> <label>: returns 0 on COMPLETE, 1 on ERROR/CANCEL
  local s
  while true; do
    s=$(status "$1")
    case "$s" in
      COMPLETE) say "$2: COMPLETE"; return 0;;
      ERROR|CANCEL*) say "❌ $2: $s"; return 1;;
    esac
    say "$2: ${s:-status unavailable}"
    sleep 300
  done
}

say "▶ pipeline v5 started"
wait_for georgian-translate-full-run "translation 20k" || exit 1

rm -rf data/qgen/translate-full && mkdir -p data/qgen/translate-full
kg kernels output nikakatsitadze/georgian-translate-full-run -p data/qgen/translate-full > /dev/null 2>&1
[ -f data/qgen/translate-full/translations.parquet ] || { say "❌ translations.parquet missing"; exit 1; }
say "translation info: $(tr -d '\n ' < data/qgen/translate-full/translate_info.json)"

built=$(uv run python qgen/build_en_pairs.py data/qgen/translate-full/translations.parquet 2>&1) || { say "❌ build_en_pairs failed: $built"; exit 1; }
say "pairs: ${built%%$'\n'*}"

kg datasets create -p data/kaggle/train-pairs-v5 2>&1 | tail -1 | sed "s/^/$(date +%H:%M) | upload: /"
for i in {1..60}; do
  ds=$(perl -e 'alarm 60; exec @ARGV' uv run kaggle datasets status nikakatsitadze/georgian-embedding-train-pairs-v5 2>&1 | tail -1)
  [[ "$ds" == *ready* ]] && break
  sleep 30
done
[[ "$ds" == *ready* ]] || { say "❌ dataset not ready: $ds"; exit 1; }
say "dataset ready"

kg kernels push -p kaggle/teacher_pairs_v5 2>&1 | tail -1 | sed "s/^/$(date +%H:%M) | /"
sleep 60
wait_for georgian-teacher-pairs-v5 "teacher scoring v5" || exit 1

kg kernels push -p kaggle/rank_v5 2>&1 | tail -1 | sed "s/^/$(date +%H:%M) | /"
sleep 60
wait_for georgian-rank-v5 "training v5" || exit 1

mkdir -p data/models/rank-v5
kg kernels output nikakatsitadze/georgian-rank-v5 -p data/models/rank-v5 > /dev/null 2>&1
[ -d data/models/rank-v5/student-v5-epoch1 ] || { say "❌ student-v5-epoch1 missing"; exit 1; }
say "v5 info: $(python3 -c "import json; d=json.load(open('data/models/rank-v5/rank_v5_info.json')); print(d['epochs'])")"

say "benchmark v5 started"
uv run python baseline.py --models data/models/rank-v5/student-v5-epoch1 > results/bench_v5.log 2>&1 || { say "❌ benchmark failed"; exit 1; }
say "🏁 benchmark: $(grep -E 'student-v5-epoch1 ' results/baseline.md)"
