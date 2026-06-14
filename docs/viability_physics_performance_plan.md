# Viability Physics Evaluation Performance Plan

## Goal

Make physics-backed viability evaluations fast enough to support DOE, active
learning, feasible-policy search, and final verification without relying on an
internal phase surrogate unless it is still justified after direct evaluation is
measured and optimized.

## Current Evidence

- Candidate-policy batches are embarrassingly parallel. The viability evaluator
  already dispatches independent policies through `ProcessPoolExecutor` when
  `workers > 1`.
- Initial small full-force, one-year probe over 8 policies measured:
  - `workers=1`: 23.8 s wall time
  - `workers=4`: 8.6 s wall time
  - `workers=8`: 5.8 s wall time
- After the first allocator optimization pass, the same 8-policy probe measured:
  - `workers=1`: 12.9 s wall time
  - `workers=4`: 5.0 s wall time
  - `workers=8`: 3.4 s wall time
- A one-squadron, one-year profile showed the single-policy physics path is
  dominated by Python allocation overhead, not heavy numerical kernels:
  repeated pilot scans, candidate-list construction, sorting, and calls to
  `rules.can_fill_seat`.
- The first optimization pass reduced the profiled single-design time from
  about 8.6 s to about 4.8 s for the benchmarked one-year design.

## Execution Slices

1. **Benchmark and profile harness**
   - Add a reusable script that can benchmark direct physics viability batches
     across worker counts.
   - Include cProfile support for one representative single-policy solve.
   - Keep outputs under ignored `outputs/`.
   - Status: complete in `tools/benchmark_viability_physics.py` and
     `src/viability/performance.py`.

2. **Parallel evaluation ergonomics**
   - Make CLI/help/docs make the existing parallel batch path obvious.
   - Guard invalid worker counts clearly.
   - Add tests proving CLI worker overrides are passed through.
   - Status: complete. `configs/viability.example.yaml` now defaults to
     `run.workers: 4`, invalid worker counts fail clearly, and `run-doe`
     reports the effective worker count.

3. **Single-policy allocator optimization**
   - Start with the hottest safe slice: reduce repeated eligibility scans in
     syllabus, continuation-training, and sim-RAP allocation.
   - Preserve event-level behavior with focused tests before broad rewrites.
   - Status: first pass complete. Assignment now uses `min` instead of sorting
     full candidate lists, CT round-robin caches static eligibility pools by
     required qualification, and deterministic zero-noise allocation uses fast
     direct attribute keys.
   - Note: `allocation_noise=0` no longer consumes random numbers while building
     assignment keys. This removes meaningless RNG burn, but exact seeded
     trajectories can shift slightly when later model steps use randomness.

4. **Re-benchmark and decide**
   - Re-run the direct physics benchmark after each implementation slice.
   - Update this document with measured before/after timings.
   - Decide whether the GPR surrogate is still needed for search throughput, or
     only for dashboard guidance and uncertainty-aware screening.
   - Status: complete for the first pass. Direct physics is now fast enough for
     small DOE and candidate-verification batches when run with workers, but
     broader active-learning campaigns should still be sized from measured
     throughput rather than assumed surrogate necessity.

## Current Recommendation

- Use physics-backed direct evaluation with `--workers` as the baseline for DOE,
  search verification, and timing comparisons.
- Do not justify a replacement surrogate from the old serial direct-solve timing.
  Re-run any surrogate-vs-direct decision against the post-optimization direct
  benchmark.
- Keep the outer GPR surrogate for dashboard guidance, uncertainty-aware
  screening, and local feasible-bound recommendations unless a full direct
  search proves cheap enough.
- If more speed is needed, the next focused engineering slice should optimize
  event-capacity checks and CT assignment with maintained per-pilot event loads
  or heap/lazy-priority queues. The final profile still spends most time in
  `allocate_continuation_training`, `assign_sortie`, `_eligible_for_event`, and
  `Pilot.has_events_capacity`.

## Non-Goals For This Slice

- Do not rewrite the whole phase simulator in NumPy until the smaller allocator
  optimizations are measured.
- Do not push branch changes.
- Do not track generated benchmark outputs.
