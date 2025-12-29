#include <math.h>

void kernel(const int   *row_ptr,
          const int   *col_ind,
          const float *val,
          const float *B,   // dense [K*N], row-major
          float       *C,   // dense [M*N], row-major
          int M, int N)
{
    // C = 0
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i*N + j] = 0.0f;
        }
    }

    // for each nonzero A(i,k): C(i,:) += A(i,k) * B(k,:)
    for (int i = 0; i < M; ++i) {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            int   k = col_ind[p];
            float a = val[p];

            for (int j = 0; j < N; ++j) {
                C[i*N + j] = fmaf(a, B[k*N + j], C[i*N + j]);
            }
        }
    }
}