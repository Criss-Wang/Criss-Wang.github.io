bfloat16

## choice of parameter data structure

DL-specific Data structure
Bfloat16

1. Motivation:
   - Ensure identical behavior for underflows, overflows, and NaNs -> bfloat16 has the same exponent size as FP32.
   - However, bfloat16 handles denormals differently from FP32: it flushes them to zero.
   - Unlike FP16, which typically requires special handling via techniques such as loss scaling, BF16 comes close to being a drop-in replacement for FP32 when training and running deep neural networks.
