# [MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators(SC24)](https://dl.acm.org/doi/pdf/10.1109/SC41406.2024.00040)


## Background

Memory-Bound && Compute-Intensive Operators, like Self-attention operators, contributed only 11%, 14%, and 19% of the FLOPs for sequence lengths of 512, 1024, and 2048, respectively, they disproportionately dominate execution time, consuming 39%, 51%, and 61% of it.

## [Source code](https://zenodo.org/records/10971908), based on Ansor V14, and Triton 2.1.

Code changes: **Incomplete**
1. 


## Features
1. New search space?
2. Reduce redundant memory access.
3. Pruning?
4. Analytical model?
5. Fusion.

## Evaluation

### Performance and tuning time

A100 and RTX3080

1. GEMM and self-attention Operators: better than Pytorch(Triton backend?) and Ansor.

2. End-to-End models(Bert-Small, Bert-Base, and Bert-Large): MCFuser + Ansor achieved about a 1.3× speedup over Ansor. MCFuser+Relay achieved about a 1.5x speedup over Relay alone.

4. Tuning time: For single operators, tuning takes 40 seconds. For end-to-end model like Bert, tuning spans hours and can be at most 50% faster than Ansor's tuning time(around 4hours).

### Effectiveness of the System Design

1. Model efficiency: correlation and figures.

2. Shared Memory Estimation: figures.


## Notes

* Good for Matmul and self-attention, not yet work for convolution.

* Add Triton as a baseline: "Finally, the PTX code produced by Triton is converted into the TVM runtime library using the ```runtime.module.loadfile_ptx``` interface and encapsulated within an ```OperatorModule``` for evaluation."


