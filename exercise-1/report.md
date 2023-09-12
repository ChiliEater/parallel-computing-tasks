---
Author: Jonas Costa
Header:
  Center: ""
  Right: "{{ChangeDate}}"
---

# TDT4200 Problem Set 1

## What is the von Neumann bottleneck? Give one example of how we can try to overcome the von Neumann bottleneck.

Programs made for von Neumann architectures essentially become read-modify-write cycles. This has the consequence that the processor will be working at the speed of memory which is usually slower.

Things that may help:

- Caching certainly helps in reducing reads and deferring writes.
- Find an alternative to von Neumann that doesn't completely break existing programs.
- Force parallelization into von Neumann programs.

## Look at the code below. Does it contain a data, name or control dependency? Justify your answer.

```c
int a[100], b[100];

for (int i = 0; i < 100; i++) {
    b[i] = i;
    a[i] = 2*b[i];
    b[i] = i*i;
}
```

We have a data dependence as `b[i]` is first assigned and then used as an input. A good compiler might be able to optimize this dependency away by replacing the input with `i`.

We then have a name dependency as the result of `b[i]` is used in `a` and is then overwritten.

We can't start working on the next iteration as `i` is used inside the loop while also being the loop condition. Therefore we have a control dependency.


## What is the difference between SIMD and MIMD in Flynn’s taxonomy?

SIMD allows us to run an instruction an multiple operands. Everything still happens in sequence.

MIMD enables us to process multiple instructions on multiple operands simultaneously.

<div style="break-after:page"></div>

## Collect and document the execution time of the integration loop when using the different solvers (Jacobi, Gauss-Seidel, and Red-black Gauss-Seidel). Comment on the amount of computational work vs. the execution time for the different solvers.

**Jacobi**: 3'869'716 iterations, 53.961263 seconds, pretty fast but needs lots of iterations.

**Gauss-Seidel**: 3'079'497 iterations, 73.391418 seconds, a lot slower but takes way less iterations.

**Red-Black Gauss-Seidel**: 2'826'014 iterations, 37.995942 seconds, very fast and less iterations!

## Find and document the model name of your CPU. You can use the commands below.

Intel(R) Core(TM) i7-8705G CPU @ 3.10GHz