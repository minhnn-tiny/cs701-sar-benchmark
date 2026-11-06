#!/usr/bin/env bash
# phase2.sh (run from the repo root): once Phase 2 is open and dataset/test/ is complete, predict and check the two test
# zips the user chose on 1 Nov (seed 6 and seed 0 of YASP350, each with SWA; the fused decoding). It does not upload:
# that is lb_upload.sh, by hand, after a look at both zips (Phase 2 allows 2 scored uploads in all).
LB=https://web-production-c241e.up.railway.app
S=workspace/mtdet/submissions; R=workspace/mtdet/runs; D=o2m_nms0.8_a0.5_k100_t0_f1

ready() {  # phase 2 open, and every image of dataset/test/images.json on disk
    curl -sS "$LB/api/status" 2>/dev/null | .venv/bin/python -c \
        "import json, sys; d = json.load(sys.stdin); sys.exit(0 if d['phase'] == 2 and d['state'] == 'open' else 1)" \
        2>/dev/null || return 1
    [ -f dataset/test/images.json ] || return 1
    .venv/bin/python -c "
import json, sys
from pathlib import Path
images = json.loads(Path('dataset/test/images.json').read_text())['images']
sys.exit(0 if images and all((Path('dataset/test') / im['file_name']).is_file() for im in images) else 1)" 2>/dev/null
}

until ready; do sleep 60; done
sleep 60  # an extraction may still be writing the last files
ready || { echo "dataset/test changed while settling; run again"; exit 1; }
echo "$(date '+%F %T') phase 2 open; dataset/test has $(.venv/bin/python -c "import json; print(len(json.load(open('dataset/test/images.json'))['images']))") images"
for r in YASP350_s6_swa YASP350_swa; do
    workspace/gpu_run.sh --min-free 8000 -- env HF_HUB_OFFLINE=1 .venv/bin/python workspace/mtdet/predict.py \
        $R/${r}_avgbn --split test --decoding $D --out $S/${r}_f1_test.zip 2>&1 | grep -v Warning | tail -n 9
    .venv/bin/python workspace/sarmtl/check_submission.py $S/${r}_f1_test.zip --split test | tail -n 1
    ls -l $S/${r}_f1_test.zip | awk '{print $5, "bytes"}'
done
