# Paired retrieval diagnostics

This bundle contains CPU utilities for paired occlusion attribution, exact four-cell branch substitution, and deterministic identity-disjoint partition planning, and budget-matched selective replay. It does not redistribute images, identities, embeddings, weights, or query-level outputs.

From this directory, install the listed NumPy dependency in your existing Python environment and run:

```bash
python -m unittest test_week2_partition test_occlusion_error_attribution test_occlusion_branch_factorial test_week35_selective_replay
```

The tests use synthetic inputs only. For an authorized local directory, see [the partition protocol](diagnostics/week2_partition_protocol.md). The ranking utilities take aligned score matrices in memory; an outer runner is responsible for validating checkpoint ancestry, role-only mounts, sample order, and native-score parity. These tools do not establish that a particular reranker improves accuracy.

Selective replay is a protocol utility, not a trained or validated improvement. It accepts the calibration role only and fails closed on gate failure.
