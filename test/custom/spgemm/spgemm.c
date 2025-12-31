
#include <math.h>
void kernel(
    // Matrix A in CSR format (M x K)
    const int   *row_ptr_A,   // size M+1
    const int   *col_ind_A,   // size nnz_A
    const float *val_A,       // size nnz_A
    // Matrix B in CSR format (K x N)
    const int   *row_ptr_B,   // size K+1
    const int   *col_ind_B,   // size nnz_B
    const float *val_B,       // size nnz_B
    // Output matrix C (dense, M x N)
    float       *C,           // size M*N, row-major
    int M, int N, int K)
{
    // Initialize C to zero
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i*N + j] = 0.0f;
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int pa = row_ptr_A[i]; pa < row_ptr_A[i + 1]; ++pa) {
            int k = col_ind_A[pa];       // column index in A = row index in B
            float a_ik = val_A[pa];      // value A(i, k)

            for (int pb = row_ptr_B[k]; pb < row_ptr_B[k + 1]; ++pb) {
                int j = col_ind_B[pb];   // column index in B
                float b_kj = val_B[pb];  // value B(k, j)

                C[i*N + j] = fmaf(a_ik, b_kj, C[i*N+j]);
            }
        }
    }
}
