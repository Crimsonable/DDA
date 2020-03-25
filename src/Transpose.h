#pragma once
#include "forwardDecleration.h"
#include "Pack.h"
#include "memory.h"
#include "Matrix.h"
#include "SIMD.h"

#ifdef _DEBUG
#include "testTools.h"
#endif // _DEBUG

using std::enable_if;
using std::is_same_v;
#define MAT_SCALAR_INDICATOR(MAT_TYPE,SCALAR_TYPE) is_same_v<typename internal::traits<##MAT_TYPE##>::scalar,##SCALAR_TYPE##>


namespace DDA {
	inline void transpose8_8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
		__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
		__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
		__t0 = _mm256_unpacklo_ps(row0, row1);
		__t1 = _mm256_unpackhi_ps(row0, row1);
		__t2 = _mm256_unpacklo_ps(row2, row3);
		__t3 = _mm256_unpackhi_ps(row2, row3);
		__t4 = _mm256_unpacklo_ps(row4, row5);
		__t5 = _mm256_unpackhi_ps(row4, row5);
		__t6 = _mm256_unpacklo_ps(row6, row7);
		__t7 = _mm256_unpackhi_ps(row6, row7);
		__tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
		__tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
		__tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
		__tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
		__tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
		__tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
		__tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
		__tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
		row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
		row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
		row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
		row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
		row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
		row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
		row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
		row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
	}

	inline void transpose8_8_ps(float* mat, float* matT, const int& leftrows, const int& leftcols, const int& T_rows, const int& T_cols) {

		__m256 r0, r1, r2, r3, r4, r5, r6, r7;
		__m256  t0, t1, t2, t3, t4, t5, t6, t7;
		int m1 = 1 % leftcols, m2 = 2 % leftcols, m3 = 3 % leftcols,
			m4 = 4 % leftcols, m5 = 5 % leftcols, m6 = 6 % leftcols, m7 = 7 % leftcols;
		int n1 = 1 % leftrows, n2 = 2 % leftrows, n3 = 3 % leftrows,
			n4 = 4 % leftrows, n5 = 5 % leftrows, n6 = 6 % leftrows, n7 = 7 % leftrows;

		r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0 * 8 + 0])), _mm_load_ps(&mat[m4 * 8 + 0]), 1);
		r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[m1 * 8 + 0])), _mm_load_ps(&mat[m5 * 8 + 0]), 1);
		r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[m2 * 8 + 0])), _mm_load_ps(&mat[m6 * 8 + 0]), 1);
		r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[m3 * 8 + 0])), _mm_load_ps(&mat[m7 * 8 + 0]), 1);
		r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0 * 8 + 4])), _mm_load_ps(&mat[m4 * 8 + 4]), 1);
		r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[m1 * 8 + 4])), _mm_load_ps(&mat[m5 * 8 + 4]), 1);
		r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[m2 * 8 + 4])), _mm_load_ps(&mat[m6 * 8 + 4]), 1);
		r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[m3 * 8 + 4])), _mm_load_ps(&mat[m7 * 8 + 4]), 1);

		t0 = _mm256_unpacklo_ps(r0, r1);
		t1 = _mm256_unpackhi_ps(r0, r1);
		t2 = _mm256_unpacklo_ps(r2, r3);
		t3 = _mm256_unpackhi_ps(r2, r3);
		t4 = _mm256_unpacklo_ps(r4, r5);
		t5 = _mm256_unpackhi_ps(r4, r5);
		t6 = _mm256_unpacklo_ps(r6, r7);
		t7 = _mm256_unpackhi_ps(r6, r7);

		r0 = _mm256_shuffle_ps(t0, t2, 0x44);
		r1 = _mm256_shuffle_ps(t0, t2, 0xEE);
		r2 = _mm256_shuffle_ps(t1, t3, 0x44);
		r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
		r4 = _mm256_shuffle_ps(t4, t6, 0x44);
		r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
		r6 = _mm256_shuffle_ps(t5, t7, 0x44);
		r7 = _mm256_shuffle_ps(t5, t7, 0xEE);

		if (leftcols!=8) {
			Mask<v_256<float>> mask;
			int step_rows = 8 - leftcols;
			for (int i = 0; i < leftcols; ++i)
				mask.d[i] = ~0;
			store_mask(matT + n7 * T_rows, mask.v, r7);
			store_mask(matT + n6 * T_rows, mask.v, r6);
			store_mask(matT + n5 * T_rows, mask.v, r5); 
			store_mask(matT + n4 * T_rows, mask.v, r4); 
			store_mask(matT + n3 * T_rows, mask.v, r3); 
			store_mask(matT + n2 * T_rows, mask.v, r2); 
			store_mask(matT + n1 * T_rows, mask.v, r1); 
			store_mask(matT, mask.v, r0); 
		}
		else {			
			_mm256_store_ps(&matT[n7 * T_rows], r7);
			_mm256_store_ps(&matT[n6 * T_rows], r6);
			_mm256_store_ps(&matT[n5 * T_rows], r5);
			_mm256_store_ps(&matT[n4 * T_rows], r4);
			_mm256_store_ps(&matT[n3 * T_rows], r3);
			_mm256_store_ps(&matT[n2 * T_rows], r2);
			_mm256_store_ps(&matT[n1 * T_rows], r1);
			_mm256_store_ps(&matT[0 * T_rows], r0);		
		}
	}

	template<typename Mat, typename MatT, typename enable_if<MAT_SCALAR_INDICATOR(Mat,float) && MAT_SCALAR_INDICATOR(MatT,float),int>::type=0>
	inline void transpose(Mat *mat, MatT *matT) {
		int rows = mat->rows;
		int cols = mat->cols;
		float *mat_ptr = mat->data();
		float *matT_ptr = matT->data();
		int EndVec_rows = rows % 8 ? rows + 8 - rows % 8 : rows;
		int EndVec_cols = cols % 8 ? cols + 8 - cols % 8 : cols;
		float *buffer = mynew<float>(EndVec_rows * 8, VECTORIZATION_ALIGN_BYTES);

#ifdef DEBUG_INFO
		DEBUG_TOOLS::printRawMatrix(mat_ptr, rows, cols, "Mat: ");
#endif // _DEBUG

		for (int j = 0; j < cols; j += 8) {
			int leftcols = cols - j >= 8 ? 8 : cols - j;
			PackMatrix_T(mat_ptr + j * rows, 8, leftcols, rows, cols, buffer, 64);
#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(buffer, EndVec_rows, 8, "buffer: ");
#endif // _DEBUG
#pragma omp parallel
			{
#pragma omp for schedule(static) nowait
				for (int i = 0; i < EndVec_rows; i += 8) {
					int leftrows = rows - i >= 8 ? 8 : rows - i;
					transpose8_8_ps(buffer + i * leftcols, matT_ptr + i * cols + j, leftrows, leftcols, cols, rows);
				}
			}
		}
		aligned_free(buffer);
	}
}