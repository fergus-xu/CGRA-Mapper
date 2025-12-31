// COO
//  void kernel(int rows, int K,
//               const float * __restrict vals,
//               const int   * __restrict cols,
//               const float * __restrict x,
//               float       * __restrict y)
// {
//   for (int r = 0; r < rows; r++) {
//     float acc = 0.0f;

//     const float *v = vals + (long)r * K;
//     const int   *c = cols + (long)r * K;

//     #if defined(__clang__)
//       #pragma clang loop vectorize(enable) interleave(enable)
//     #elif defined(__GNUC__)
//       #pragma GCC ivdep
//     #endif
//     for (int j = 0; j < K; j++) {
//       acc += v[j] * x[c[j]];
//     }

//     y[r] += acc;
//   }
// }

// DIA

// void kernel(int nrows,
//               int ndiags,
//               const int   *doff,
//               const float *av,
//               const float *x,
//               float       *y)
// {
//   for (int i = 0; i < nrows; i++) {
//     float sum = 0.0f;

//     // Inner loop over diagonals
// #if defined(__clang__)
//     #pragma clang loop vectorize(enable) interleave(enable)
// #elif defined(__GNUC__)
//     #pragma GCC ivdep
// #endif
//     for (int j = 0; j < ndiags; j++) {
//       int k = i + doff[j];

//       if (k >= 0 && k < nrows) {
//         sum += av[i * ndiags + j] * x[k];
//       }
//     }

//     y[i] = sum;
//   }
// }

// CSR
void kernel(int nrows,
            const int   *row,
            const int   *col,
            const float *v,
            const float *x,
            float       *y)
{
  for (int i = 0; i < nrows; i++) {
    float sum = 0.0f;

    int start = row[i];
    int end   = row[i + 1];

    for (int j = start; j < end; j++) {
      sum += v[j] * x[col[j]];
    }

    y[i] = sum;
  }
}