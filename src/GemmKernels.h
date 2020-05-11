#pragma once
#include "forwardDecleration.h"
#include "SIMD.h"
#include "memory.h"

#define ELEMENT(i, j, k, p) c_##i##j##_c_##k##j##_vreg.d[p]
#define VELEMENT(i, j, k) c_##i##j##_c_##k##j##_vreg.v

namespace CSM
{
	using namespace CSM::SSE_OP;
	template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
	FORCE_INLINE void VEC_CALL AddDot8x1(T *a, T *b, T *c, int block_k, int m, int n, int k, int lda, int ldb, int current_c_col, int padRows, bool padMode)
	{
		int p;
		constexpr int VSIZE = sizeof(Vtype) / sizeof(T);

		Vtype
			c_00_c_30_vreg,c_40_c_70_vreg,
			a_0p_a_3p_vreg, a_4p_a_7p_vreg,
			b_p0_vreg;

		T *b_pntr, *a_top_pntr, *a_down_pntr;
		b_pntr = b;

		a_top_pntr = a;
		a_down_pntr = a + VSIZE;

		load_ps(VELEMENT(0, 0, 3), c);
		load_ps(VELEMENT(4, 0, 7), c + VSIZE);

		for (int p = 0; p < block_k; ++p) {
			load_ps(a_0p_a_3p_vreg.v, a_top_pntr);
			load_ps(a_4p_a_7p_vreg.v, a_down_pntr);

			load_ps1(b_p0_vreg.v, b_pntr);
			b_pntr += ldb;

			c_00_c_30_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p0_vreg.v, c_00_c_30_vreg.v);
			c_40_c_70_vreg.v = fmadd(a_4p_a_7p_vreg.v, b_p0_vreg.v, c_40_c_70_vreg.v);

			a_top_pntr += lda;
			a_down_pntr += lda;
		}

		if (padMode)
		{
			Mask<Vtype> mask;
			if (padRows > VSIZE) {
				for (int i = 0; i < VSIZE - padRows % VSIZE; ++i)
					mask.d[i] = ~0;
				store_mask(c, mask.v, VELEMENT(0, 0, 3));
			}
			else {
				for (int i = 0; i < VSIZE - padRows; ++i)
					mask.d[i] = ~0;
				store(c, VELEMENT(0, 0, 3));
				store_mask(c + VSIZE, mask.v, VELEMENT(4, 0, 7));
			}
		}
		else
		{
			store(c, VELEMENT(0, 0, 3));
			store(c + VSIZE, VELEMENT(4, 0, 7));
		}
	}

	template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
	FORCE_INLINE void AddDot4x4(T *a, T *b, T *c, const int &block_k, const int &m, const int &n, const int &k, const int &current_c_col, const int &padRows, bool padMode)
	{
		constexpr int VSIZE = sizeof(Vtype) / sizeof(T);
		const int left_cols = n - current_c_col > 4 ? 4 : n - current_c_col;
		std::size_t c1 = 1 % left_cols, c2 = 2 % left_cols, c3 = 3 % left_cols;

		Vtype
			c_00_c_30_vreg,
			c_01_c_31_vreg, c_02_c_32_vreg, c_03_c_33_vreg,
			a_0p_a_3p_vreg,
			b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

		T *b_pntr;
		b_pntr = b;

		if (!m % 4)
		{
			load_ps(VELEMENT(0, 0, 3), c);
			load_ps(VELEMENT(0, 1, 3), c + c1 * m);
			load_ps(VELEMENT(0, 2, 3), c + c2 * m);
			load_ps(VELEMENT(0, 3, 3), c + c3 * m);
		}
		else
		{
			loadu_ps(VELEMENT(0, 0, 3), c);
			loadu_ps(VELEMENT(0, 1, 3), c + c1 * m);
			loadu_ps(VELEMENT(0, 2, 3), c + c2 * m);
			loadu_ps(VELEMENT(0, 3, 3), c + c3 * m);
		}

		for (int p = 0; p < block_k; ++p)
		{
			load_ps(a_0p_a_3p_vreg.v, a + VSIZE * p);

			load_ps1(b_p0_vreg.v, b_pntr++);
			load_ps1(b_p1_vreg.v, b_pntr++);
			load_ps1(b_p2_vreg.v, b_pntr++);
			load_ps1(b_p3_vreg.v, b_pntr++); /* load and duplicate */

#ifdef CPUID__AVX2__
			c_00_c_30_vreg.v = a_0p_a_3p_vreg.v * b_p0_vreg.v + c_00_c_30_vreg.v;
			c_01_c_31_vreg.v = a_0p_a_3p_vreg.v * b_p1_vreg.v + c_01_c_31_vreg.v;
			c_02_c_32_vreg.v = a_0p_a_3p_vreg.v * b_p2_vreg.v + c_02_c_32_vreg.v;
			c_03_c_33_vreg.v = a_0p_a_3p_vreg.v * b_p3_vreg.v + c_03_c_33_vreg.v;
#endif // CPUID__AVX2__

#ifdef CPUID__FAM__
			c_00_c_30_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p0_vreg.v, c_00_c_30_vreg.v);
			c_01_c_31_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p1_vreg.v, c_01_c_31_vreg.v);
			c_02_c_32_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p2_vreg.v, c_02_c_32_vreg.v);
			c_03_c_33_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p3_vreg.v, c_03_c_33_vreg.v);
#endif // CPUID__FAM__
		}

		if (padMode)
		{
			Mask<Vtype> mask;
			for (int i = 0; i < VSIZE - padRows; ++i)
				mask.d[i] = ~0;
			store_mask(c + c3 * m, mask.v, VELEMENT(0, 3, 3));
			store_mask(c + c2 * m, mask.v, VELEMENT(0, 2, 3));
			store_mask(c + c1 * m, mask.v, VELEMENT(0, 1, 3));
			store_mask(c, mask.v, VELEMENT(0, 0, 3));
		}
		else
		{
			store(c + c2 * m, VELEMENT(0, 2, 3));
			store(c + c3 * m, VELEMENT(0, 3, 3));
			store(c + c1 * m, VELEMENT(0, 1, 3));
			store(c, VELEMENT(0, 0, 3));
		}
	}

	template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
	FORCE_INLINE void VEC_CALL AddDot8x4(T *a, T *b, T *c, int block_k, int n, const int& ldc, const int& lda, const int& ldb, int current_c_col, int padRows, bool padMode)
	{
		int p;
		const int left_cols = n - current_c_col > 4 ? 4 : n - current_c_col;
		constexpr int VSIZE = sizeof(Vtype) / sizeof(T);
		std::size_t c1 = 1 % left_cols, c2 = 2 % left_cols, c3 = 3 % left_cols;

		Vtype
			c_00_c_30_vreg,
			c_01_c_31_vreg, c_02_c_32_vreg, c_03_c_33_vreg,
			c_40_c_70_vreg, c_41_c_71_vreg, c_42_c_72_vreg, c_43_c_73_vreg,
			a_0p_a_3p_vreg, a_4p_a_7p_vreg,
			b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

		T *b_pntr, *a_top_pntr, *a_down_pntr;
		b_pntr = b;

		a_top_pntr = a;
		a_down_pntr = a + VSIZE;

		load_ps(VELEMENT(0, 0, 3), c);
		load_ps(VELEMENT(0, 1, 3), c + c1 * ldc);
		load_ps(VELEMENT(0, 2, 3), c + c2 * ldc);
		load_ps(VELEMENT(0, 3, 3), c + c3 * ldc);
		load_ps(VELEMENT(4, 0, 7), c + VSIZE);
		load_ps(VELEMENT(4, 1, 7), c + VSIZE + c1 * ldc);
		load_ps(VELEMENT(4, 2, 7), c + VSIZE + c2 * ldc);
		load_ps(VELEMENT(4, 3, 7), c + VSIZE + c3 * ldc);

		for (p = 0; p < block_k; p++)
		{
			load_ps(a_0p_a_3p_vreg.v, a_top_pntr);
			load_ps(a_4p_a_7p_vreg.v, a_down_pntr);

			load_ps1(b_p0_vreg.v, b_pntr);
			b_pntr += ldb;
			load_ps1(b_p1_vreg.v, b_pntr);
			b_pntr += ldb;
			load_ps1(b_p2_vreg.v, b_pntr);
			b_pntr += ldb;
			load_ps1(b_p3_vreg.v, b_pntr);
			b_pntr += ldb;					/* load and duplicate */

			//_mm_prefetch((char*)(b_p0_pntr + 1), _MM_HINT_T0);

#ifdef CPUID__AVX2__
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
#endif // CPUID__AVX2__

#ifdef CPUID__FAM__
			/* 0-3 rows */
			c_00_c_30_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p0_vreg.v, c_00_c_30_vreg.v);
			c_01_c_31_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p1_vreg.v, c_01_c_31_vreg.v);
			c_02_c_32_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p2_vreg.v, c_02_c_32_vreg.v);
			c_03_c_33_vreg.v = fmadd(a_0p_a_3p_vreg.v, b_p3_vreg.v, c_03_c_33_vreg.v);

			/* 4-7 rows */
			c_40_c_70_vreg.v = fmadd(a_4p_a_7p_vreg.v, b_p0_vreg.v, c_40_c_70_vreg.v);
			c_41_c_71_vreg.v = fmadd(a_4p_a_7p_vreg.v, b_p1_vreg.v, c_41_c_71_vreg.v);
			c_42_c_72_vreg.v = fmadd(a_4p_a_7p_vreg.v, b_p2_vreg.v, c_42_c_72_vreg.v);
			c_43_c_73_vreg.v = fmadd(a_4p_a_7p_vreg.v, b_p3_vreg.v, c_43_c_73_vreg.v);
#endif // CPUID__FAM__

			a_top_pntr += lda;
			a_down_pntr += lda;
		}

		if (padMode)
		{
			Mask<Vtype> mask;
			if (padRows > VSIZE) {
				for (int i = 0; i < VSIZE - padRows % VSIZE; ++i)
					mask.d[i] = ~0;
				store_mask(c + c3 * ldc, mask.v, VELEMENT(0, 3, 3));
				store_mask(c + c2 * ldc, mask.v, VELEMENT(0, 2, 3));
				store_mask(c + c1 * ldc, mask.v, VELEMENT(0, 1, 3));
				store_mask(c, mask.v, VELEMENT(0, 0, 3));
			}
			else {
				for (int i = 0; i < VSIZE - padRows; ++i)
					mask.d[i] = ~0;
				store(c + c3 * ldc, VELEMENT(0, 3, 3));
				store_mask(c + VSIZE + c3 * ldc, mask.v, VELEMENT(4, 3, 7));
				store(c + c2 * ldc, VELEMENT(0, 2, 3));
				store_mask(c + VSIZE + c2 * ldc, mask.v, VELEMENT(4, 2, 7));
				store(c + c1 * ldc, VELEMENT(0, 1, 3));
				store_mask(c + VSIZE + c1 * ldc, mask.v, VELEMENT(4, 1, 7));
				store(c, VELEMENT(0, 0, 3));
				store_mask(c + VSIZE, mask.v, VELEMENT(4, 0, 7));
			}
		}
		else
		{
			store(c + c3 * ldc, VELEMENT(0, 3, 3));
			store(c + VSIZE + c3 * ldc, VELEMENT(4, 3, 7));
			store(c + c2 * ldc, VELEMENT(0, 2, 3));
			store(c + VSIZE + c2 * ldc, VELEMENT(4, 2, 7));
			store(c + c1 * ldc, VELEMENT(0, 1, 3));
			store(c + VSIZE + c1 * ldc, VELEMENT(4, 1, 7));
			store(c, VELEMENT(0, 0, 3));
			store(c + VSIZE, VELEMENT(4, 0, 7));
		}
	}

	FORCE_INLINE void VEC_CALL AddDot8x8_ps(float *a, float *b, float *c, int block_k, int m, int n, int k, int current_c_col, int padRows, bool padMode)
	{
		__m256 c_00_c_70_vreg, c_01_c_71_vreg, c_02_c_72_vreg, c_03_c_73_vreg,
			c_04_c_74_vreg, c_05_c_75_vreg, c_06_c_76_vreg, c_07_c_77_vreg,
			a_0p_a_7p_vreg,
			b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg, b_p4_vreg, b_p5_vreg, b_p6_vreg, b_p7_vreg;

		const int left_cols = n - current_c_col > 8 ? 8 : n - current_c_col;
		std::size_t c1 = 1 % left_cols, c2 = 2 % left_cols, c3 = 3 % left_cols, c4 = 4 % left_cols,
			c5 = 5 % left_cols, c6 = 6 % left_cols, c7 = 7 % left_cols;
		float *a_ptr = a, *b_ptr = b;

		load_ps(c_00_c_70_vreg, c);
		load_ps(c_01_c_71_vreg, c + c1 * m);
		load_ps(c_02_c_72_vreg, c + c2 * m);
		load_ps(c_03_c_73_vreg, c + c3 * m);
		load_ps(c_04_c_74_vreg, c + c4 * m);
		load_ps(c_05_c_75_vreg, c + c5 * m);
		load_ps(c_06_c_76_vreg, c + c6 * m);
		load_ps(c_07_c_77_vreg, c + c7 * m);

		for (int p = 0; p < block_k; ++p)
		{
			load_ps(a_0p_a_7p_vreg, a_ptr);

			load_ps(b_p0_vreg, b_ptr);
			b_p1_vreg = _mm256_permute_ps(b_p0_vreg, 0x01);

#ifdef CPUID__AVX2__
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
#endif // CPUID__AVX2__

#ifdef CPUID__FAM__
			/* 0-3 rows */
			c_00_c_70_vreg = fmadd(a_0p_a_7p_vreg, b_p0_vreg, c_00_c_70_vreg);
			c_01_c_71_vreg = fmadd(a_0p_a_7p_vreg, b_p1_vreg, c_01_c_71_vreg);
			c_02_c_72_vreg = fmadd(a_0p_a_7p_vreg, b_p2_vreg, c_02_c_72_vreg);
			c_03_c_73_vreg = fmadd(a_0p_a_7p_vreg, b_p3_vreg, c_03_c_73_vreg);

			/* 4-7 rows */
			c_04_c_74_vreg = fmadd(a_0p_a_7p_vreg, b_p4_vreg, c_04_c_74_vreg);
			c_05_c_75_vreg = fmadd(a_0p_a_7p_vreg, b_p5_vreg, c_05_c_75_vreg);
			c_06_c_76_vreg = fmadd(a_0p_a_7p_vreg, b_p6_vreg, c_06_c_76_vreg);
			c_07_c_77_vreg = fmadd(a_0p_a_7p_vreg, b_p7_vreg, c_07_c_77_vreg);
#endif // CPUID__FAM__

			a_ptr += 8;
		}

		if (padMode)
		{
			Mask<v_256<float>> mask;
			for (int i = 0; i < 8 - padRows; ++i)
				mask.d[i] = ~0;
			store_mask(c + c7 * m, mask.v, c_07_c_77_vreg);
			store_mask(c + c6 * m, mask.v, c_06_c_76_vreg);
			store_mask(c + c5 * m, mask.v, c_05_c_75_vreg);
			store_mask(c + c4 * m, mask.v, c_04_c_74_vreg);
			store_mask(c + c3 * m, mask.v, c_03_c_73_vreg);
			store_mask(c + c2 * m, mask.v, c_02_c_72_vreg);
			store_mask(c + c1 * m, mask.v, c_01_c_71_vreg);
			store_mask(c, mask.v, c_00_c_70_vreg);
		}
		else
		{
			store(c + c7 * m, c_07_c_77_vreg);
			store(c + c6 * m, c_06_c_76_vreg);
			store(c + c5 * m, c_05_c_75_vreg);
			store(c + c4 * m, c_04_c_74_vreg);
			store(c + c3 * m, c_03_c_73_vreg);
			store(c + c2 * m, c_02_c_72_vreg);
			store(c + c1 * m, c_01_c_71_vreg);
			store(c, c_00_c_70_vreg);
		}
	}


	FORCE_INLINE void VEC_CALL Gemm_kernel_avx256(float* a, const int& lda, float* b, const int& ldb, float* c, const int& ldc, const int& k,int leftRows,int leftCols)
	{
		__m256 a0, a1, b0, b1, b2, b3;
		__m256 c0, c1, c2, c3, c4, c5, c6, c7;
		int m1 = 1 % leftCols, m2 = 2 % leftCols, m3 = 3 % leftCols;

		float *b_pntr, *a_top_pntr, *a_down_pntr;
		b_pntr = b;

		a_top_pntr = a;
		a_down_pntr = a + 8;

		c0 = _mm256_load_ps(c);
		c1 = _mm256_load_ps(c + 8);
		c2 = _mm256_load_ps(c + m1 * ldc);
		c3 = _mm256_load_ps(c + m1 * ldc + 8);
		c4 = _mm256_load_ps(c + m2 * ldc);
		c5 = _mm256_load_ps(c + m2 * ldc + 8);
		c6 = _mm256_load_ps(c + m3 * ldc);
		c7 = _mm256_load_ps(c + m3 * ldc + 8);

		for (int p = 0; p < k; ++p)
		{
			a0 = _mm256_load_ps(a_top_pntr);
			a1 = _mm256_load_ps(a_down_pntr);
			b0 = _mm256_broadcast_ss(b_pntr);
			b_pntr += ldb;
			b1 = _mm256_broadcast_ss(b_pntr);
			b_pntr += ldb;
			b2 = _mm256_broadcast_ss(b_pntr);
			b_pntr += ldb;
			b3 = _mm256_broadcast_ss(b_pntr);
			b_pntr += ldb;					/* load and duplicate */

#ifdef CPUID__FAM__
			/* 0-3 rows */
			c0 = _mm256_fmadd_ps(a0, b0, c0);
			c2 = _mm256_fmadd_ps(a0, b1, c2);
			c4 = _mm256_fmadd_ps(a0, b2, c4);
			c6 = _mm256_fmadd_ps(a0, b3, c6);
			c1 = _mm256_fmadd_ps(a1, b0, c1);
			c3 = _mm256_fmadd_ps(a1, b1, c3);
			c5 = _mm256_fmadd_ps(a1, b2, c5);
			c7 = _mm256_fmadd_ps(a1, b3, c7);
#endif // CPUID__FAM__

			a_top_pntr += lda;
			a_down_pntr += lda;
		}

		if (leftRows!=16)
		{
			__m256i mask = _mm256_setzero_si256();
			if (leftRows > 8) {
				for (int i = 0; i < leftRows - 8; ++i) *((int*)(&mask) + i) = -1;
				_mm256_store_ps(c + m3 * ldc, c6);
				_mm256_maskstore_ps(c + m3 * ldc + 8, mask, c7);
				_mm256_store_ps(c + m2 * ldc, c4);
				_mm256_maskstore_ps(c + m2 * ldc + 8, mask, c5);
				_mm256_store_ps(c + m1 * ldc, c2);
				_mm256_maskstore_ps(c + m1 * ldc + 8, mask, c3);
				_mm256_store_ps(c, c0);
				_mm256_maskstore_ps(c + 8, mask, c1);
			}
			else {
				for (int i = 0; i < leftRows; ++i) *((int*)(&mask) + i) = -1;;
				_mm256_maskstore_ps(c + m3 * ldc, mask, c6);
				_mm256_maskstore_ps(c + m2 * ldc, mask, c4);
				_mm256_maskstore_ps(c + m1 * ldc, mask, c2);
				_mm256_maskstore_ps(c, mask, c0);
			}
		}
		else
		{
			_mm256_store_ps(c + m3 * ldc + 8, c7);
			_mm256_store_ps(c + m3 * ldc, c6);
			_mm256_store_ps(c + m2 * ldc + 8, c5);
			_mm256_store_ps(c + m2 * ldc, c4);
			_mm256_store_ps(c + m1 * ldc + 8, c3);
			_mm256_store_ps(c + m1 * ldc, c2);
			_mm256_store_ps(c + 8, c1);
			_mm256_store_ps(c, c0);
		}
	}
} // namespace CSM
