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
- A small full-force, one-year probe over 8 policies measured:
  - `workers=1`: 23.8 s wall time
  - `workers=4`: 8.6 s wall time
  - `workers=8`: 5.8 s wall time
- A one-squadron, one-year profile showed the single-policy physics path is
  dominated by Python allocation overhead, not heavy numerical kernels:
  repeated pilot scans, candidate-list construction, sorting, and calls to
  `rules.can_fill_seat`.

## Execution Slices

1. **Benchmark and profile harness**
   - Add a reusable script that can benchmark direct physics viability batches
     across worker counts.
   - Include cProfile support for one representative single-policy solve.
   - Keep outputs under ignored `outputs/`.

2. **Parallel evaluation ergonomics**
   - Make CLI/help/docs make the existing parallel batch path obvious.
   - Guard invalid worker counts clearly.
   - Add tests proving CLI worker overrides are passed through.

3. **Single-policy allocator optimization**
   - Start with the hottest safe slice: reduce repeated eligibility scans in
     syllabus, continuation-training, and sim-RAP allocation.
   - Preserve event-level behavior with focused tests before broad rewrites.

4. **Re-benchmark and decide**
   - Re-run the direct physics benchmark after each implementation slice.
   - Update this document with measured before/after timings.
   - Decide whether the GPR surrogate is still needed for search throughput, or
     only for dashboard guidance and uncertainty-aware screening.

## Non-Goals For This Slice

- Do not rewrite the whole phase simulator in NumPy until the smaller allocator
  optimizations are measured.
- Do not push branch changes.
- Do not track generated benchmark outputs.
