# asmpt

![asmpt output](https://raw.githubusercontent.com/fgarlin/asmpt/master/output.png)

A small toy path tracer written in x86 assembly, inspired by [smallpt](https://www.kevinbeason.com/smallpt/).

## Assemble and run

This program is intended to be used under Linux and assembled using NASM. You will also need a system that supports SSE 4.1 instructions. The path tracer prints the image to standard output, so you need to pipe it to an image file.

``` sh
make && ./asmpt > image.pbm
```

## Notes

### Floating point operations

SSE instructions are used to perform floating-point arithmetic. SSE 4.1 is required because the program uses the dot product instruction `dpps`.

### Registers

The program makes heavy use of registers to reduce memory accesses. Some registers are considered "global", i.e. their value is considered to be valid during the entire lifetime of the program:

- `r14`: PCG state
- `r15`: PCG seq
- `xmm8`: per-pixel accumulated pixel color
- `xmm9`: per-pixel throughput
- `xmm12`: u vector (camera right)
- `xmm13`: v vector (camera up)
- `xmm14`: w vector (camera forward)
- `xmm15`: w' vector

### RNG

The [PCG random number generator](https://www.pcg-random.org/index.html) is used to generate random samples for Monte Carlo.

### Trigonometric functions

The sine function is approximated by a look-up table with 1024 entries for angles between 0 and π/2. The sine (and cosine) for the rest of the angles are deduced by symmetry.

## License

This code is released under the MIT license. See LICENSE for more details.
