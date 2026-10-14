# Figure 8(a): warm-up sensitivity by number of historical starts

Scope: Complete the left panel of Figure 8 with $k \in \{1,2,4,8\}$, exactly the same set as Figure 8(b), without changing the selected $k=8$ reference algorithm.

- Workflow: [GeoIVF Actions #38516776145](https://github.com/gilga1983/GeoIVF/actions/runs/38516776145), self-hosted runner, workflow at .github/workflows/navhints-warmup-starts.yml.
- Experiment code: scripts/qualify_final_core_warmup_starts.py and scripts/run_final_core_warmup_starts_ci.sh.
- Frozen inputs: pinned Microsoft DiskANN fcf90534174cf29c78c9f13b4cccf1fcabff85f5; PubMed1M/MedCPT; first 5K disjoint training queries, next 5K causally replayed; 14,464 junction IDs, 2,048 global FIFO destination IDs, 512 regions, no persistent hints.
- History lengths: 0,250,500,1000,2000,3000,4000 causal requests; exactly 500 measured requests per suffix. Each k is compared against the same window's junction-only reference at six mutually reachable Recall@10 targets (35,45,55,65,72,75%) without extrapolation.
- Upon successful run: resulting JSON member warmup-starts/final-core-warmup-starts.json and individual h{history}-starts{k}.log files in the workflow artifact.
- Original k=8 source: [GeoIVF #38137378869](https://github.com/gilga1983/GeoIVF/actions/runs/38137378869), artifact 11707480968; compare the core values in figures/data/eval_warmup_final_core.csv in the paper repository.
- The three relevant original NavHints code patch blob hashes are identical between original experiment source a7e20bd236eb0abd5394d4c91056e9f944b222a4 and current GeoIVF main: 427d52360d59493680bd6a0f2c6a003b7598e51f; 2cd0b5a2cbd8ea059cdd1dc0745a9679d2f6f6eb; e99fa8d6fc22393553a1197bf0d669041b8e1083.
- Do NOT reuse the four curves from Figure 8(b) for 8(a): that is a different experiment varying the demand population, with a 4K/1K warm-up/test split.
- Paper update gate: confirmed successful real-SSD workflow and artifact; all seven history windows and four k arms; six joint recall anchors per point; zero persistent writes; new k=8 logical-I/O read savings reproducible against the original k=8 curve. Only then add new CSV with history,k1,k2,k4,k8, change Figure 8(a) to four curves using the same styles as panel (b), and refresh only its caption and interpretation. Do not change Figure 8(b) or other results.
- Results are descriptive replays of one query stream rather than independent demand samples.
