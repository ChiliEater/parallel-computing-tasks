---
Author: Jonas Costa
Header:
  Center: ""
  Right: "{{ChangeDate}}"
---

# TDT4200 Problem Set 5

## In which scenarios could a CUDA implementation outperform a CPU-based implementation?

Problems that can be massively parallelized are optimal for GPUs. This assignment is a pretty good example.

## In which scenarios could a CPU-based implementation outperform a CUDA implementation?

CPUs outperform GPUs whenever sequential tasks come up as CPUs usually have higher clocks than GPUs. DRAM usually also has higher clock speeds than VRAM which helps in load-store-heavy operations.

## Implementation Note

For some reason my output has some issues in the top and right sides of the domain. I'm not sure what could be causing this as I:

- allocate the usual amount ($(N + 2) * (M + 2)$)
- round up my grid size
- stop all threads that would be outside the domain (larger than $N + 1$ or $M + 1$)
- calculate the boundary condition in the same way as the sequential one (at least I think I do)

As a result, I'm not really sure what is causing this weird behaviour. I *am* getting a compiler warning that reads as follows: `warning: narrowing conversion of "(int)ceil(long int)(M / 32)" from "int" to unsigned "int"`. Maybe it has something to do with that? Removing the cast doesn't seem to help.

<div style="break-after:page"></div>
