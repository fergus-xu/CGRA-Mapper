clang-21 -emit-llvm -O3 -fno-unroll-loops -o kernel.bc -c spmm.c
llvm-dis-21 kernel.bc -o kernel.ll
opt-21 -passes='loop-unroll' -unroll-count=4 -o kernel_unroll.bc kernel.bc
llvm-dis-21 kernel_unroll.bc -o kernel_unroll.ll
