__kernel void copy(__global float *a,
                  __global float *b) {
   *b = *a;
}

__kernel void add(__global float *a,
                  __global float *b,
                  __global float *c) {
   *c = *a + *b;
}

__kernel void sub(__global float *a,
                  __global float *b,
                  __global float *c) {
   *c = *a - *b;
}

__kernel void mult(__global float *a,
                   __global float *b,
                   __global float *c) {
   *c = *a * *b;
}

__kernel void div(__global float *a,
                  __global float *b,
                  __global float *c) {
   *c = *a / *b;
}

__kernel void incre(__global float *a,
                  __global float *b) {
   *b = *a + 1;
}

__kernel void decre(__global float *a,
                  __global float *b) {
   *b = *a - 1;
}