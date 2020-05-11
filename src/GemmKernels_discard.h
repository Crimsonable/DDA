#pragma once
#include "SIMD.h"


namespace GemmCpuKernels {
	void kernel_256_avx(float* a, int lda, float* b, int ldb, float* c, int ldc, int leftrows = 8, int leftcols = 8) {
		__m256 ar, bn, c0, c1, c2, c3, c4, c5, c6, c7;
		int m1 = 1 % leftcols, m2 = 2 % leftcols, m3 = 3 % leftcols,
			m4 = 4 % leftcols, m5 = 5 % leftcols, m6 = 6 % leftcols, m7 = 7 % leftcols;

		c0 = _mm256_load_ps(c);
		c1 = _mm256_load_ps(c + m1 * ldc);
		c2 = _mm256_load_ps(c + m2 * ldc);
		c3 = _mm256_load_ps(c + m3 * ldc);
		c4 = _mm256_load_ps(c + m4 * ldc);
		c5 = _mm256_load_ps(c + m5 * ldc);
		c6 = _mm256_load_ps(c + m6 * ldc);
		c7 = _mm256_load_ps(c + m7 * ldc);

#pragma unroll
		for (int i = 0; i < 8; ++i) {
			const int offset = i * 8;
			ar = _mm256_load_ps(a + i * lda);

			bn = _mm256_broadcast_ss(b + offset);
			c0 = _mm256_fmadd_ps(ar, bn, c0);
			bn = _mm256_broadcast_ss(b + offset + ldb);
			c1 = _mm256_fmadd_ps(ar, bn, c1);
			bn = _mm256_broadcast_ss(b + offset + 2 * ldb);
			c2 = _mm256_fmadd_ps(ar, bn, c2);
			bn = _mm256_broadcast_ss(b + offset + 3 * ldb);
			c3 = _mm256_fmadd_ps(ar, bn, c3);
			bn = _mm256_broadcast_ss(b + offset + 4 * ldb);
			c4 = _mm256_fmadd_ps(ar, bn, c4);
			bn = _mm256_broadcast_ss(b + offset + 5 * ldb);
			c5 = _mm256_fmadd_ps(ar, bn, c5);
			bn = _mm256_broadcast_ss(b + offset + 6 * ldb);
			c6 = _mm256_fmadd_ps(ar, bn, c6);
			bn = _mm256_broadcast_ss(b + offset + 7 * ldb);
			c7 = _mm256_fmadd_ps(ar, bn, c7);
		}
		if (leftrows != 8) {
			__m256i mask = _mm256_setzero_si256();
			for (int i = 0; i < leftrows; ++i) *((int*)(&mask) + i) = -1;
			_mm256_maskstore_ps(c + m7 * ldc, mask, c7);
			_mm256_maskstore_ps(c + m6 * ldc, mask, c6);
			_mm256_maskstore_ps(c + m5 * ldc, mask, c5);
			_mm256_maskstore_ps(c + m4 * ldc, mask, c4);
			_mm256_maskstore_ps(c + m3 * ldc, mask, c3);
			_mm256_maskstore_ps(c + m2 * ldc, mask, c2);
			_mm256_maskstore_ps(c + m1 * ldc, mask, c1);
			_mm256_maskstore_ps(c, mask, c0);
		}
		else {
			_mm256_store_ps(c + m7 * ldc, c7);
			_mm256_store_ps(c + m6 * ldc, c6);
			_mm256_store_ps(c + m5 * ldc, c5);
			_mm256_store_ps(c + m4 * ldc, c4);
			_mm256_store_ps(c + m3 * ldc, c3);
			_mm256_store_ps(c + m2 * ldc, c2);
			_mm256_store_ps(c + m1 * ldc, c1);
			_mm256_store_ps(c, c0);
		}
	}

	void kernel_256_avx(float* a, int lda, float* b, int ldb, 
						__m256& c0, __m256& c1, __m256& c2, __m256& c3, __m256& c4, __m256& c5, __m256& c6, __m256& c7) {

		__m256 ar, bn;

#pragma unroll
		for (int i = 0; i < 8; ++i) {
			const int offset = i * 8;
			ar = _mm256_load_ps(a + i * lda);

			bn = _mm256_broadcast_ss(b + offset);
			c0 = _mm256_fmadd_ps(ar, bn, c0);
			bn = _mm256_broadcast_ss(b + offset + ldb);
			c1 = _mm256_fmadd_ps(ar, bn, c1);
			bn = _mm256_broadcast_ss(b + offset + 2 * ldb);
			c2 = _mm256_fmadd_ps(ar, bn, c2);
			bn = _mm256_broadcast_ss(b + offset + 3 * ldb);
			c3 = _mm256_fmadd_ps(ar, bn, c3);
			bn = _mm256_broadcast_ss(b + offset + 4 * ldb);
			c4 = _mm256_fmadd_ps(ar, bn, c4);
			bn = _mm256_broadcast_ss(b + offset + 5 * ldb);
			c5 = _mm256_fmadd_ps(ar, bn, c0);
			bn = _mm256_broadcast_ss(b + offset + 6 * ldb);
			c6 = _mm256_fmadd_ps(ar, bn, c6);
			bn = _mm256_broadcast_ss(b + offset + 7 * ldb);
			c7 = _mm256_fmadd_ps(ar, bn, c7);
		}
	}
}