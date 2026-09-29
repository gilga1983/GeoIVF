#!/usr/bin/env python3
"""Run upstream MQSim against one exported fixture trace; archive all inputs."""
import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoivf.mqsim import export
p = argparse.ArgumentParser()
p.add_argument('--mqsim', default='third_party/MQSim')
p.add_argument('--trace', default='artifacts/smoke/geopack-combined.jsonl')
p.add_argument('--out', default='artifacts/mqsim')
a = p.parse_args()
source = Path(a.mqsim).resolve()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
config = ET.parse(source/'ssdconfig.xml')
# Small integration device, not a calibrated commercial SSD model.
changes = dict(Flash_Channel_Count='2', Chip_No_Per_Channel='1', Die_No_Per_Chip='1',
               Plane_No_Per_Die='2', Block_No_Per_Plane='256', Page_No_Per_Block='128',
               Page_Capacity='4096', IO_Queue_Depth='64', Queue_Fetch_Size='16')
for tag, value in changes.items():
    node = config.find('.//'+tag)
    if node is None:
        raise ValueError(f'MQSim configuration missing {tag}')
    node.text = value
config.write(out/'ssd.xml')
report = export(a.trace, out, request_gap_ns=100000, ssd_config=out/'ssd.xml')
with (out/'console.log').open('w') as log:
    subprocess.run([str(source/'MQSim'), '-i', str(out/'ssd.xml'), '-w', str(out/'workload.xml')],
                   cwd=out, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                   check=True, timeout=180)
outputs = list(out.glob('workload_scenario_*.xml'))
if not outputs:
    raise RuntimeError('MQSim did not produce a scenario result')
report['mqsim_output'] = [str(f) for f in outputs]
report['mqsim_commit'] = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                                text=True).strip()
report['calibration'] = 'integration-only device; not commercial NVMe calibration'
(out/'integration.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
