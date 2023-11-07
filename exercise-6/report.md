---
Author: Jonas Costa
Header:
  Center: ""
  Right: "{{ChangeDate}}"
---

# TDT4200 Problem Set 6

## What are the limitations of using cooperative groups to sync the whole grid at once?

The following limitations apply to cooperative groups when the whole grid is used:
- All the blocks in the grid must fit onto the GPU at the same time → size constraints
- Calling such kernel is a bit more verbose
- Early returns are not an option anymore
- Thread divergence can be an issue, though this is also generally the case

<div style="break-after:page"></div>
