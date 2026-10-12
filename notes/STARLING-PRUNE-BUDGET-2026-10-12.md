# NavHints for Starling: prune and resize existing components

**Status:** experimental branch; baseline and paper are frozen. No new
heuristics, route gates, frontier logic, graph changes, or index-layout changes.

## Question

The portable DiskANN configuration B=14464, C=2048, k=8 can waste CPU in
original Starling when its own memory navigator supplies strong entry points.
Determine whether deleting the underperforming component and reducing the
remaining scoring budgets recovers latency at matched Recall@10.

## Stage 1: Remove components

Compare on the same original native Starling graph, native metric, and query
split, with every disabled component's PQ scoring actually bypassed:

- baseline: native Starling, no NavHints (B0,C0,k0).
- final_core: frozen portable NavHints (B14464,C2048,k8).
- junction_only: B14464,C0,k0.
- destination_only: B0,C2048,k8.

## Stage 2: Small budgets (after interpreting stage 1)

- junction_compact: B4096,C0,k0.
- destination_compact: B0,C512,k2.
- joint_compact: B4096,C512,k2.
- Repeat the native baseline, frozen portable NavHints, and both component-only
  controls in the same session for comparable native latency measurements.

512 geometric regions and eight probed regions are retained for either junction
directory; this keeps the junction heuristic unchanged and reduces the number
of scored IDs. The compact destination mode changes only C and k. No warm-up,
query label, or held-out result is used to choose the smaller configurations.

## Source/reproducibility contract

- Native author Starling pinned SHA:
  17dc3e8a011533a62374445f53963e951b72883a.
- The source adapters run in a temporary author checkout, building on the
  validated final-Core patch. Main and the paper remain untouched.
- The full B14464 and compact B4096 directories derive from the same disjoint
  initial 5K training queries; online histories are unique FIFO and updated
  strictly after search completion.
- Coveo preserves 5K training / 20K chronological warm-up / 5K measurement.
  PubMed, Text2Image-10M and BigANN-10M preserve 5K / 4K / 1K.
- PubMed and Text2Image retain float32 MIPS; BigANN native uint8 L2; Coveo
  retains its validated normalized float-native search.
- Three rotated repetitions, same SSD/graph per workload and native navigator.
- Primary metric: native search latency at matched Recall@10. Secondary:
  provider logical SSD reads, query CPU time, and accounted hint payload.
- Exact same source/metric/query workload, no cross-engine speed ratios.
- Report target recall anchors only within jointly observed recall ranges; do
  not extrapolate. Experiments in this note are ablations. Selecting an
  architecture-aware champion requires a disjoint validation prefix and
  confirmation on the held-out suffix.
- This investigation does not change the frozen selected NavHints design, paper
  prose, or competitor measurements until a validated result is approved.

## Implementation

- scripts/patch_starling_prune_budget.py: guarded native C++ changes.
- scripts/generate_starling_prune_budget.py: the source-fidelity generator for
  all four datasets and both stages.
- scripts/summarize_starling_prune_budget.py: strict 3-repetition matched-recall
  report for all ablation arms.
- .github/workflows/starling-prune-budget.yml: guarded source checks followed by
  serialized original-Starling self-hosted native pilot runs.

Original frozen evidence: native Starling workflow 38137155521 (Coveo/BigANN),
workflow 38186111308 (PubMed/Text2Image). Earlier source-level complementary
hints and gating experiments are archived separately, and are outside this
deliberately narrow pruning experiment.
