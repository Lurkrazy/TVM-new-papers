# [MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators(SC24)](https://dl.acm.org/doi/pdf/10.1109/SC41406.2024.00040)


## Background

Memory-Bound && Compute-Intensive Operators, like Self-attention operators, contributed only 11%, 14%, and 19% of the FLOPs for sequence lengths of 512, 1024, and 2048, respectively, they disproportionately dominate execution time, consuming 39%, 51%, and 61% of it.


## Features
1. New search space(much smaller).
2. Codegen optimization(reduce redundant memory access)
3. Pruning.(same as 1.)
4. Analytical model.
5. Fusion.

## Evaluation

### Performance and tuning time

A100 and RTX3080

1. GEMM and self-attention Operators: better than Pytorch(Triton backend?) and Ansor.

2. End-to-End models(Bert-Small, Bert-Base, and Bert-Large):  + Ansor achieved about a 1.3× speedup over Ansor. MCFuser+Relay achieved about a 1.5x speedup over Relay alone.

4. Tuning time: For single operators, tuning takes 40 seconds. For an end-to-end model like Bert, tuning spans hours and can be at most 50% faster than Ansor's tuning time(around 4 hours).

### Effectiveness of the System Design

1. Model efficiency: correlation and figures.

2. Shared Memory Estimation: figures.


## Notes

* My understanding is they convert the TIR to TritonIR. Process: Relay IR -> (with tvm.tir.Schedule primitives) converted to tiling expression+TIR -> (with TIR visitor and IR translator) converted to TritonIR. (TODO: read tir2triton.py)

* Good for Matmul and self-attention, but it has not yet worked for convolution. As Triton doesn't contain high-performance implementation for convolution(pytorch+cudnn is 2x faster), maybe it's not a good direction.

* Add Triton as a baseline: "Finally, the PTX code produced by Triton is converted into the TVM runtime library using the `runtime.module.loadfile_ptx` interface and encapsulated within an `OperatorModule` for evaluation."


## [Source code](https://zenodo.org/records/10971908), based on Ansor V14, and Triton 2.1.

### Code changes in TVM:

#### In python\tvm\contrib\mcfuser

Need to look at `auto_search_meta.py`

1. `utils.py`
2. `tir2triton.py`
3. `test_helpers.py`
4. `partition.py`
5. `mcschedule.py`
6. `d2lutils.py`
7. `d2ltir.py`
8. `d2lschedule.py`
9. `d2ldriver.py`
10. `d2ldoc.py`
11. `config.py`
12. `compile.py`
13. `common.py`
14. `build.py`
15. `auto_search_meta.py`

#### In src/relay/backend/contrib/mcfuser

Reused the code from `src/relay/backend/contrib/cutlass`. They added a new compilation pass `CompileForMCFuser()` for MCFuser, used for integrating the MCFuser module into the TVM compilation pipeline, allowing custom compiled code to be generated and executed as part of the TVM framework. 

(I assume it utilizes Triton for code generation?)

1. `codegen.cc`
2. `codegen.h`
3. `target.cc`

