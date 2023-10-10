---
Author: Jonas Costa
Header:
  Center: ""
  Right: "{{ChangeDate}}"
---

# TDT4200 Problem Set 3

## What is a critical section?

Critical sections are specific code sections where execution may not overlap. Ignoring this requirement leads to undefined behaviour.

## How can a critical section be protected from race conditions?

There some options available. Here are just the ones mentioned in the lecture slides:

- Atomic operations; These cannot overlap
- Load Linked & Store Conditional
- Mutexes and semaphores
- Constructs from higher-level languages

## What is oversubscription?

Oversubscription is simply creating more threads than the amount of available compute units. In some cases this allows compute units to be busy for longer as oversubscribed threads can use CPU time when another thread is waiting.

## Problem:

*Suppose that you have a set of n pthreads, each with a local integer that represents each thread’s individual index from 0 through n − 1, and a print statement at the end of its function body. How would you implement a method to ensure that the output of the print statements appears in the order of the thread indices? A pseudo-code description is sufficient to answer this question.*

```rust
// We assume that we have the thread's index
let mut thread: u32 = █;
// Thread zero does not need to wait
if thread > 0 {
    join(thread - 1); // Wait for previous thread
}
println!("Thread {} here!", thread);
return 0;

```

Of course, this could also be extracted into a seperate function with the thread index as an argument.

<div style="break-after:page"></div>