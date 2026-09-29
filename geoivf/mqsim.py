"""Open-loop device workload export, NOT a closed-loop ANN latency simulator."""
from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def export(trace_path, out_dir, *, request_gap_ns: int, ssd_config):
    """Give ALL methods the same stated request-spacing policy.

The source JSONL keeps query/stage dependencies, but vanilla MQSim's five
columns cannot express them. Output is deliberately marked open-loop and
cannot be used as an end-to-end query latency/QPS measurement.
"""
    if request_gap_ns < 1:
        raise ValueError('request_gap_ns must be positive and explicit')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line]
    if not records:
        raise ValueError('empty trace')
    config = ET.parse(ssd_config)
    def count(tag):
        e = config.find('.//'+tag)
        if e is None or e.text is None or int(e.text) < 1:
            raise ValueError(f'missing/invalid SSD geometry: {tag}')
        return int(e.text)
    channels = count('Flash_Channel_Count')
    chips = count('Chip_No_Per_Channel')
    dies = count('Die_No_Per_Chip')
    planes = count('Plane_No_Per_Die')
    # Reject address wraparound in the single-flow experiment.
    sectors = (channels*chips*dies*planes*count('Block_No_Per_Plane')
               *count('Page_No_Per_Block')*count('Page_Capacity'))//512
    op = config.find('.//Overprovisioning_Ratio')
    if op is not None:
        sectors = int(sectors*(1-float(op.text)))
    trace = out/'requests.trace'
    total = 0
    with trace.open('w') as f:
        for i, r in enumerate(records):
            offset, size = r['offset_bytes'], r['length_bytes']
            if r['operation'] != 'read' or offset < 0 or size < 1 or offset%512 or size%512:
                raise ValueError('MQSim requires nonnegative, sector-aligned READ requests')
            if (offset+size)//512 > sectors:
                raise ValueError('trace exceeds modeled SSD capacity; refusing LBA wrapping')
            f.write(f'{(i+1)*request_gap_ns} 0 {offset//512} {size//512} 1\n')
            total += size
    root = ET.Element('MQSim_IO_Scenarios')
    scenario = ET.SubElement(root, 'IO_Scenario')
    flow = ET.SubElement(scenario, 'IO_Flow_Parameter_Set_Trace_Based')
    fields = dict(Priority_Class='HIGH', Device_Level_Data_Caching_Mode='TURNED_OFF',
                  Channel_IDs=','.join(map(str, range(channels))),
                  Chip_IDs=','.join(map(str, range(chips))),
                  Die_IDs=','.join(map(str, range(dies))),
                  Plane_IDs=','.join(map(str, range(planes))),
                  Initial_Occupancy_Percentage='50', File_Path=str(trace.resolve()),
                  Percentage_To_Be_Executed='100', Relay_Count='1', Time_Unit='NANOSECOND')
    for key, value in fields.items():
        ET.SubElement(flow, key).text = value
    ET.indent(root)
    ET.ElementTree(root).write(out/'workload.xml', encoding='utf-8', xml_declaration=True)
    report = dict(mode='open-loop-device-only', request_gap_ns=request_gap_ns,
                  requests=len(records), read_bytes=total, sector_bytes=512,
                  read_opcode=1, query_latency_valid=False,
                  source_trace=str(Path(trace_path).resolve()),
                  warning='Fixed arrivals ignore stage completion dependencies. '
                          'Do not report MQSim IOPS as ANN QPS or sum request latencies.')
    (out/'export.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
