#pragma once
#include "forwardDecleration.h"
#include "SIMD.h"
#include "memory.h"

#define ELEMENT(i, j, k, p) c_##i##j##_c_##k##j##_vreg.d[p]
#define VELEMENT(i,j,k) c_##i##j##_c_##k##j##_vreg.v

namespace DDA {
	using namespace DDA::SSE_OP;
	template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
	inline void AddDot8x1(T *a, T *b, T *c, int block_k, int m, int n, int k) {
		const constexpr int VSIZE = sizeof(Vtype) / sizeof(T);
		Vtype
			c_00_c_30_vreg,
			c_40_c_70_vreg,
			a_0p_a_3p_vreg, a_4p_a_7p_vreg,
			b_p0_vreg;

		T *b_p0_pntr = b;
		const constexpr int step = 1;
		for (int p = 0; p < block_k; p += step) {
			load_ps(a_0p_a_3p_vreg.v, a + VSIZE * 2 * p);
			load_ps(a_4p_a_7p_vreg.v, a + VSIZE * (2 * p + 1));

			load_ps1(b_p0_vreg.v, b_p0_pntr++);
#ifdef CPUID__AVX2__
			c_00_c_30_vreg.v = b_p0_vreg.v * a_0p_a_3p_vreg.v + c_00_c_30_vreg.v;
			c_40_c_70_vreg.v = b_p0_vreg.v * a_4p_a_7p_vreg.v + c_40_c_70_vreg.v;
#endif // CPUID__AVX2__

#ifdef CPUID__FAM__
			c_00_c_30_vreg.v = fmadd(b_p0_vreg.v, a_0p_a_3p_vreg.v, c_00_c_30_vreg.v);
			c_40_c_70_vreg.v = fmadd(b_p0_vreg.v, a_4p_a_7p_vreg.v, c_00_c_30_vreg.v);
#endif // CPUID__FAM__

		}

		for (int i = 0; i < VSIZE; ++i) {
			*(c + i) += ELEMENT(0, 0, 3, i);
			*(c + VSIZE + i) += ELEMENT(4, 0, 7, i);
		}
	}

	template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
	inline void AddDot4x1(T *a, T *b, T *c, int block_k, int m, int n, int k) {
		const constexpr int VSIZE = sizeof(Vtype) / sizeof(T);
		Vtype
			c_00_c_30_vreg,
			a_0p_a_3p_vreg,
			b_p0_vreg;

		T *b_p0_pntr = b;
		const constexpr int step = 1;
		for (int p = 0; p < block_k; p += step) {
			load_ps(a_0p_a_3p_vreg.v, a + VSIZE * p);
			load_ps1(b_p0_vreg.v, b_p0_pntr++);
#ifdef CPUID__AVX2__
			c_00_c_30_vreg.v = b_p0_vreg.v * a_0p_a_3p_vreg.v + c_00_c_30_vreg.v;
#endif // CPUID__AVX2__

#ifdef CPUID__FAM__
			c_00_c_30_vreg.v = fmadd(b_p0_vreg.v, a_0p_a_3p_vreg.v, c_00_c_30_vreg.v);
#endif // CPUID__FAM__           
		}

		for (int i = 0; i < VSIZE; ++i) {
			*(c + i) += ELEMENT(0, 0, 3, i);
		}
	}

	template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
	void AddDot4x4(T *a, T *b, T *c, int block_k, int m, int n, int k, int current_c_col, int padRows, bool padMode) {
		constexpr int VSIZE = sizeof(Vtype) / sizeof(T);
		const int left_cols = n - current_c_col > 4 ? 4 : n - current_c_col;
		std::size_t c1 = 1 % left_cols, c2 = 2 % left_cols, c3 = 3 % left_cols;

		Vtype
			c_00_c_30_vreg,
			c_01_c_31_vreg, c_02_c_32_vreg, c_03_c_33_vreg,
			a_0p_a_3p_vreg,
			b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

		T *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

		b_p0_pntr = b;
		b_p1_pntr = b + block_k;
		b_p2_pntr = b + 2 * block_k;
		b_p3_pntr = b + 3 * block_k;

		load_ps(VELEMENT(0, 0, 3), c);
		load_ps(VELEMENT(0, 1, 3), c + c1 * m);
		load_ps(VELEMENT(0, 2, 3), c + c2 * m);
		load_ps(VELEMENT(0, 3, 3), c + c3 * m);

		for (int p = 0; p < block_k; ++p) {
			load_ps(a_0p_a_3p_vreg.v, a + VSIZE * p);

			load_ps1(b_p0_vreg.v, b_p0_pntr++);
			load_ps1(b_p1_vreg.v, b_p1_pntr++);
			load_ps1(b_p2_vreg.v, b_p2_pntr++);
			load_ps1(b_p3_vreg.v, b_p3_pntr++); /* load and duplicate */

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

		if (padMode) {
			Mask<Vtype> mask;
			for (int i = 0; i < VSIZE - padRows; ++i)
				mask.d[i] = ~0;
			store_mask(c + c3 * m, mask.v, VELEMENT(0, 3, 3));
			store_mask(c + c2 * m, mask.v, VELEMENT(0, 2, 3));
			store_mask(c + c1 * m, mask.v, VELEMENT(0, 1, 3));
			store_mask(c, mask.v, VELEMENT(0, 0, 3));
		}
		else {
			store(c + c2 * m, VELEMENT(0, 2, 3));
			store(c + c3 * m, VELEMENT(0, 3, 3));
			store(c + c1 * m, VELEMENT(0, 1, 3));
			store(c, VELEMENT(0, 0, 3));			
		}
	}

	template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
	void AddDot8x4(T *a, T *b, T *c, int block_k, int m, int n, int k, int current_c_col, int padRows, bool padMode) {
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

		T *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

		b_p0_pntr = b;
		b_p1_pntr = b + block_k;
		b_p2_pntr = b + 2 * block_k;
		b_p3_pntr = b + 3 * block_k;

		load_ps(VELEMENT(0, 0, 3), c);
		load_ps(VELEMENT(0, 1, 3), c + c1 * m);
		load_ps(VELEMENT(0, 2, 3), c + c2 * m);
		load_ps(VELEMENT(0, 3, 3), c + c3 * m);
		load_ps(VELEMENT(4, 0, 7), c + VSIZE);
		load_ps(VELEMENT(4, 1, 7), c + VSIZE + c1 * m);
		load_ps(VELEMENT(4, 2, 7), c + VSIZE + c2 * m);
		load_ps(VELEMENT(4, 3, 7), c + VSIZE + c3 * m);


		for (p = 0; p < block_k; p++) {
			load_ps(a_0p_a_3p_vreg.v, a + p * VSIZE * 2);
			load_ps(a_4p_a_7p_vreg.v, a + VSIZE + p * VSIZE * 2);

			load_ps1(b_p0_vreg.v, b_p0_pntr++);
			load_ps1(b_p1_vreg.v, b_p1_pntr++);
			load_ps1(b_p2_vreg.v, b_p2_pntr++);
			load_ps1(b_p3_vreg.v, b_p3_pntr++); /* load and duplicate */
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
		}
		
		
		if (padMode) {
			Mask<Vtype> mask;
			for (int i = 0; i < VSIZE-padRows; ++i)
				mask.d[i] = ~0;
			store(c + c3 * m, VELEMENT(0, 3, 3));
			store_mask(c + VSIZE + c3 * m, mask.v, VELEMENT(4, 3, 7));
			store(c + c2 * m, VELEMENT(0, 2, 3));
			store_mask(c + VSIZE + c2 * m, mask.v, VELEMENT(4, 2, 7));
			store(c + c1 * m, VELEMENT(0, 1, 3));
			store_mask(c + VSIZE + c1 * m, mask.v, VELEMENT(4, 1, 7));
			store(c, VELEMENT(0, 0, 3));
			store_mask(c + VSIZE, mask.v, VELEMENT(4, 0, 7));	
		}
		else {
			store(c + c3 * m, VELEMENT(0, 3, 3));
			store(c + VSIZE + c3 * m, VELEMENT(4, 3, 7));
			store(c + c2 * m, VELEMENT(0, 2, 3));
			store(c + VSIZE + c2 * m, VELEMENT(4, 2, 7));
			store(c + c1 * m, VELEMENT(0, 1, 3));
			store(c + VSIZE + c1 * m, VELEMENT(4, 1, 7));
			store(c, VELEMENT(0, 0, 3));
			store(c + VSIZE, VELEMENT(4, 0, 7));
		}
	}
}