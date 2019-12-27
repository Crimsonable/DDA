#pragma once
#include "forwardDecleration.h"
#include "memory.h"
#define m_kernel 256
#define k_kernel 128
#define b_buffer_cols 512
#define _min_( i, j ) ( (i)<(j) ? (i): (j) )
#define ELEMENT(i,j,k,p) c_##i##j##_c_##k##j##_vreg.d[p]

namespace DDA {

	__m128 operator*(const __m128& l, const __m128& r) {
		return _mm_mul_ps(l, r);
	}

	__m128 operator+(const __m128& l, const __m128& r) {
		return _mm_add_ps(l, r);
	}

	template<typename T>
	union v2df_t
	{
		__m128 v;
		T d[4];


	};

	template<typename T>
	void AddDot4x4(T *a, T *b, T *c, int block_k, int m, int n, int k) {
		int p;
		v2df_t<T>
			c_00_c_30_vreg, c_01_c_31_vreg, c_02_c_32_vreg, c_03_c_33_vreg,
			c_40_c_70_vreg, c_41_c_71_vreg, c_42_c_72_vreg, c_43_c_73_vreg,
			a_0p_a_3p_vreg,a_4p_a_7p_vreg,
			b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

		T *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

		b_p0_pntr = b;
		b_p1_pntr = b + block_k;
		b_p2_pntr = b + 2 * block_k;
		b_p3_pntr = b + 3 * block_k;

		c_00_c_30_vreg.v = _mm_setzero_ps();
		c_01_c_31_vreg.v = _mm_setzero_ps();
		c_02_c_32_vreg.v = _mm_setzero_ps();
		c_03_c_33_vreg.v = _mm_setzero_ps();
		c_40_c_70_vreg.v = _mm_setzero_ps();
		c_41_c_71_vreg.v = _mm_setzero_ps();
		c_42_c_72_vreg.v = _mm_setzero_ps();
		c_43_c_73_vreg.v = _mm_setzero_ps();

		for (p = 0; p < block_k; p++) {
			a_0p_a_3p_vreg.v = _mm_load_ps(a + p * 8);
			a_4p_a_7p_vreg.v = _mm_load_ps(a + 4 + p * 8);

			b_p0_vreg.v = _mm_load_ps1(b_p0_pntr++);   /* load and duplicate */
			b_p1_vreg.v = _mm_load_ps1(b_p1_pntr++);   /* load and duplicate */
			b_p2_vreg.v = _mm_load_ps1(b_p2_pntr++);   /* load and duplicate */
			b_p3_vreg.v = _mm_load_ps1(b_p3_pntr++);   /* load and duplicate */

			/* 0-3 rows */
			c_00_c_30_vreg.v = a_0p_a_3p_vreg.v * b_p0_vreg.v + c_00_c_30_vreg.v;
			c_01_c_31_vreg.v = a_0p_a_3p_vreg.v * b_p1_vreg.v + c_01_c_31_vreg.v;
			c_02_c_32_vreg.v = a_0p_a_3p_vreg.v * b_p2_vreg.v + c_02_c_32_vreg.v;
			c_03_c_33_vreg.v = a_0p_a_3p_vreg.v * b_p3_vreg.v + c_03_c_33_vreg.v;

			/* 4-7 rows */
			c_40_c_70_vreg.v = a_4p_a_7p_vreg.v * b_p0_vreg.v + c_40_c_70_vreg.v;
			c_41_c_71_vreg.v = a_4p_a_7p_vreg.v * b_p1_vreg.v + c_41_c_71_vreg.v;
			c_42_c_72_vreg.v = a_4p_a_7p_vreg.v * b_p2_vreg.v + c_42_c_72_vreg.v;
			c_43_c_73_vreg.v = a_4p_a_7p_vreg.v * b_p3_vreg.v + c_43_c_73_vreg.v;

		}

			for (int i = 0; i < 4; ++i) {
				*(c + i) += ELEMENT(0, 0, 3, i);
				*(c + i + m) += ELEMENT(0, 1, 3, i);
				*(c + i + 2 * m) += ELEMENT(0, 2, 3, i);
				*(c + i + 3 * m) += ELEMENT(0, 3, 3, i);
				*(c + i + 4) += ELEMENT(4, 0, 7, i);
				*(c + i + 4 + m) += ELEMENT(4, 1, 7, i);
				*(c + i + 4 + 2 * m) += ELEMENT(4, 2, 7, i);
				*(c + i + 4 + 3 * m) += ELEMENT(4, 3, 7, i);
			}
	}

	template<typename T>
	void PackMatrixA(T* A, int blockrows, int m, int block_k, T* packA) {
		for (int j = 0; j < block_k; ++j) {
			T *ptr = A + j * m;
			for (int i = 0; i < blockrows; ++i) {
				*(packA + i + j * blockrows) = *(ptr + i);
			}
		}
	}

	template<typename T>
	void PackMatrixB(T* B, int blockcols, int block_k, int k, T* packB) {
		for (int i = 0; i < block_k; ++i) {
			for (int j = 0; j < blockcols; ++j) {
				*(packB + i + block_k * j) = *(B + i + k * j);
			}
		}
	}

	template<typename T>
	void InnerKernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int m, int n, int k, bool isFirst, bool isLast) {
		T *packedA = reinterpret_cast<T*>(aligned_alloc(rm*rk * sizeof(T), 16));
		static T *packedB = reinterpret_cast<T*>(aligned_alloc(rk* n * sizeof(T), 16));
		for (int j = 0; j < rn; j += 4) {
			if(isFirst) 
				PackMatrixB(b_B + j * k, 4, rk, k, packedB + j * rk);
#pragma omp parallel for
			for (int i = 0; i < rm; i += 8) {
				if (j == 0) 
					PackMatrixA(b_A + i, 8, m, rk, packedA + rk * i);
				AddDot4x4(packedA + rk * i, packedB + j * rk, C + i + j * m, rk, m, n, k);
			}
		}
		aligned_free(packedA);
		if(isLast)
			aligned_free(packedB);
	}

	template<typename lhs,typename rhs,typename dst>
	void Product(lhs *A, rhs *B, dst *C) {
		auto ptrA = A->data();
		auto ptrB = B->data();
		auto ptrC = C->data();
		int p, q, rp = k_kernel, rq = m_kernel;
		int m = A->rows, k = A->cols, n = B->cols;

		for (p = 0; p < k; p += rp) {
			rp = _min_(k - p, k_kernel);
			for (q = 0; q < m; q += rq) {
				rq = _min_(m - q, m_kernel);
				InnerKernel(ptrA + m * p + q, ptrB + p, ptrC + q, rq, rp, n, m, n, k, q == 0, p == k - k_kernel && q == m - m_kernel);
			}
		}
	}
}