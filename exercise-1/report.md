# TDT4200 Problem Set 1

## What is the von Neumann bottleneck? Give one example of how we can try to overcome the von Neumann bottleneck.


## Look at the code below. Does it contain a data, name or control dependency? Justify your answer.

```c
int a[100], b[100];

for (int i = 0; i < 100; i++) {
    b[i] = i;
    a[i] = 2*b[i];
    b[i] = i*i;
}
```



## What is the difference between SIMD and MIMD in Flynn’s taxonomy?


## Collect and document the execution time of the integration loop when using the different solvers (Jacobi, Gauss-Seidel, and Red-black Gauss-Seidel). Comment on the amount of computational work vs. the execution time for the different solvers.


## Find and document the model name of your CPU. You can use the commands below.

Intel(R) Core(TM) i7-8705G CPU @ 3.10GHz