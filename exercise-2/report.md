---
Author: Jonas Costa
Header:
  Center: ""
  Right: "{{ChangeDate}}"
---

# TDT4200 Problem Set 1

## What is non-uniform memory access (NUMA)? Mention an interconnect that could give NUMA effects.

Example: Fat trees

It means that memory access time depends on the location of data being read. Memory (or machines) that are farther away take more time to read but closer ones are faster.

## How is Gustafson's law different from Amdahl's law?

Amdahl's law assumes that throwing more CPUs at the problem shortens the time required to compute it. That means it will approach a limit. Gustafson's on the other hand assumes that we throw more work at the CPUs instead. That means the amount of work done grows with the number of CPUs and the time remains constant. No limit!

## What is the difference between weak scaling and strong scaling?

Strong scaling functions quickly start to level off when $f \neq 0$ whereas weak scaling functions remain closer to linear. They do level off but only much later.

## Get the execution time of the main loop in the provided sequential implementation and in your MPI implementation. Calculate and report the speedup and efficiency when using 2, 4, and 8 MPI processes.

2: 18.223853
4: 8.408469, speedup 40%
8: 21.270685 speedup -200%

<div style="break-after:page"></div>