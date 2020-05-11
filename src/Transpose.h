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


namespace CPU_OP {
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

	FORCE_INLINE void tran(float* mat, int ld, float* matT, int ldt) {
		__m256  r0, r1, r2, r3, r4, r5, r6, r7;
		__m256  t0, t1, t2, t3, t4, t5, t6, t7;

		r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0 * ld + 0])), _mm_load_ps(&mat[4 * ld + 0]), 1);
		r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[1 * ld + 0])), _mm_load_ps(&mat[5 * ld + 0]), 1);
		r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[2 * ld + 0])), _mm_load_ps(&mat[6 * ld + 0]), 1);
		r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[3 * ld + 0])), _mm_load_ps(&mat[7 * ld + 0]), 1);
		r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0 * ld + 4])), _mm_load_ps(&mat[4 * ld + 4]), 1);
		r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[1 * ld + 4])), _mm_load_ps(&mat[5 * ld + 4]), 1);
		r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[2 * ld + 4])), _mm_load_ps(&mat[6 * ld + 4]), 1);
		r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[3 * ld + 4])), _mm_load_ps(&mat[7 * ld + 4]), 1);

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

		_mm256_store_ps(&matT[0 * ldt], r0);
		_mm256_store_ps(&matT[1 * ldt], r1);
		_mm256_store_ps(&matT[2 * ldt], r2);
		_mm256_store_ps(&matT[3 * ldt], r3);
		_mm256_store_ps(&matT[4 * ldt], r4);
		_mm256_store_ps(&matT[5 * ldt], r5);
		_mm256_store_ps(&matT[6 * ldt], r6);
		_mm256_store_ps(&matT[7 * ldt], r7);
	}


	template<typename T>
	void transpose(T* mat, const int& ld1, T* dst, const int& ld2, const int& rows, const int& cols) {
		constexpr int step = 8 / sizeof(T) * sizeof(float);
		const int EndVecRow = rows / step * step;
		const int EndVecCol = cols / step * step;
#pragma omp parallel
		{
#pragma omp for schedule(dynamic) nowait
			for (int colIndex = 0; colIndex < EndVecCol; colIndex += step) {
				int offset1 = ld1 * colIndex;
				for (int rowIndex = 0; rowIndex < EndVecRow; rowIndex += step) {
					tran(mat + rowIndex + offset1, ld1, dst + rowIndex * ld2 + colIndex, ld2);
				}
				for (int rowIndex = EndVecRow; rowIndex < rows; ++rowIndex) {
					*(dst + rowIndex * ld2 + colIndex) = *(mat + rowIndex + offset1);
				}
			}
#pragma omp for schedule(dynamic) nowait
			for (int colIndex = EndVecCol; colIndex < cols; ++colIndex) {
				int offset1 = colIndex * ld1;
				for (int rowIndex = 0; rowIndex < rows; ++rowIndex) {
					*(dst + rowIndex * ld2 + colIndex) = *(mat + rowIndex + offset1);
				}
			}
		}
	}
}