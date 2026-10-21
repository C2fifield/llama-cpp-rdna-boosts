#!/usr/bin/env bash
tag="$1"; shift
./clihash.sh "$tag" "$@" >/dev/null 2>&1
python3 - "$tag" <<'PY'
import re,sys,hashlib
tag=sys.argv[1]
cur=None; out=[]
for l in open(f'/tmp/g4base/cli-{tag}.log',errors='replace'):
    m=re.search(r'called impl draft-mtp, hist size = (\d+), call_count = (\d+)', l)
    if m: out.append({"h":int(m.group(1)),"c":[]}); cur=out[-1]; continue
    m=re.search(r'spec\s+draft:.*pos\s+(\d+):\s*(\d+)\s*\(\s*([0-9.]+)\)', l)
    if m and cur is not None: cur["c"].append((int(m.group(1)),int(m.group(2)),m.group(3)))
for c in out: c["c"]=sorted(c["c"])
p0=[c["c"] for c in out[:1]]
print(f"{tag}: calls={len(out)} fp={hashlib.sha256(str(out).encode()).hexdigest()[:10]} call0_hist={out[0]['h'] if out else '-'} call0_pos0={p0[0][:3] if p0 else '-'}")
PY
