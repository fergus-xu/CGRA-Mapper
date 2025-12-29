#include <math.h>

void kernel(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; ++i) {
        c[i] = fmaf(a[i], b[i], c[i]);
    }
}