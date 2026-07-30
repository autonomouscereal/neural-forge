# A10 k32 cached-functional output fold

This experiment keeps the authenticated A9 k32 input fold fixed and solves a
separate output coefficient vector from the existing sealed 648-row LFM2
causal trace cache. It performs no model construction, parent forward pass,
trace recapture, training, gradient step, pruning, quantization, global map
solve, or network access.

The analytic solve completed in 118 logged chunks with zero failures. Its
advisory cached-hidden top-1 result was 584/648 (90.12%), including 140/155
holdout rows (90.32%) and 58/113 parent multi-code winners (51.33%). The A9
shared k32 route scored 551/648 and 21/113 respectively.

Manual whole-response evaluation remains authoritative and is performed by the
separate A11 split-C-in/C-out run.
