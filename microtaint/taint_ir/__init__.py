"""Lowering an instruction's taint propagation to a compilable program.

`frompcode` turns one instruction's p-code into a straight-line SSA program over
`ir`'s 64-bit instruction set, covering every output the instruction writes,
flags included.  `boolsynth` collapses the one-bit flag algebra a lifter emits
into cheapest expressions.  `exec` and `cbackend` are the backends: a flat-array
interpreter in C, and generated C compiled by a real optimiser.

Everything here is derived from p-code, so it is ISA-general by construction --
the only host-specific part is which backend runs the result.
"""
