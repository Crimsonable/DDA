#pragma once
#include "SIMD.h"
#include "forwardDecleration.h"
#include "memory.h"
#define m_kernel 256
#define k_kernel 128
#define inner_rows 16
#define inner_cols 4
#define basic_step 4
#define _min_(i, j) ((i) < (j) ? (i) : (j))
#define ELEMENT(i, j, k, p) c_##i##j##_c_##k##j##_vreg.d[p]

/*
A:	num*[m_kernel*k_kernel]		B:	  Bnum*[k_kernel*n]
	 ________  ________				 ____________________
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
								
*/

namespace DDA {
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

            c_00_c_30_vreg.v = b_p0_vreg.v * a_0p_a_3p_vreg.v + c_00_c_30_vreg.v;
            c_40_c_70_vreg.v = b_p0_vreg.v * a_4p_a_7p_vreg.v + c_40_c_70_vreg.v;
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

            c_00_c_30_vreg.v = b_p0_vreg.v * a_0p_a_3p_vreg.v + c_00_c_30_vreg.v;
        }

        for (int i = 0; i < VSIZE; ++i) {
            *(c + i) += ELEMENT(0, 0, 3, i);
        }
    }

    template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
    void AddDot4x4(T *a, T *b, T *c, int block_k, int m, int n, int k, int current_c_col) {
        constexpr int VSIZE = sizeof(Vtype) / sizeof(T);
        const int left_cols = n - current_c_col > 4 ? 4 : n - current_c_col;
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

        for (int p = 0; p < block_k; ++p) {
            load_ps(a_0p_a_3p_vreg.v, a + VSIZE * p);

            load_ps1(b_p0_vreg.v, b_p0_pntr++);
            load_ps1(b_p1_vreg.v, b_p1_pntr++);
            load_ps1(b_p2_vreg.v, b_p2_pntr++);
            load_ps1(b_p3_vreg.v, b_p3_pntr++); /* load and duplicate */

            c_00_c_30_vreg.v = a_0p_a_3p_vreg.v * b_p0_vreg.v + c_00_c_30_vreg.v;
            c_01_c_31_vreg.v = a_0p_a_3p_vreg.v * b_p1_vreg.v + c_01_c_31_vreg.v;
            c_02_c_32_vreg.v = a_0p_a_3p_vreg.v * b_p2_vreg.v + c_02_c_32_vreg.v;
            c_03_c_33_vreg.v = a_0p_a_3p_vreg.v * b_p3_vreg.v + c_03_c_33_vreg.v;
        }

        std::size_t c1 = 1 % left_cols, c2 = 2 % left_cols, c3 = 3 % left_cols;

        for (int i = 0; i < VSIZE; ++i) {
            *(c + i) += ELEMENT(0, 0, 3, i);
            *(c + i + c1 * m) += ELEMENT(0, 1, 3, i);
            *(c + i + c2 * m) += ELEMENT(0, 2, 3, i);
            *(c + i + c3 * m) += ELEMENT(0, 3, 3, i);
        }
    }

    template <typename T, typename Vtype, typename std::enable_if<std::is_same_v<Vtype, v_128<T>> || std::is_same_v<Vtype, v_256<T>>, int>::type = 0>
    void AddDot8x4(T *a, T *b, T *c, int block_k, int m, int n, int k, int current_c_col) {
        int p;
        const int left_cols = n - current_c_col > 4 ? 4 : n - current_c_col;
        constexpr int VSIZE = sizeof(Vtype) / sizeof(T);
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

        for (p = 0; p < block_k; p++) {
            load_ps(a_0p_a_3p_vreg.v, a + p * VSIZE * 2);
            load_ps(a_4p_a_7p_vreg.v, a + VSIZE + p * VSIZE * 2);

            load_ps1(b_p0_vreg.v, b_p0_pntr++);
            load_ps1(b_p1_vreg.v, b_p1_pntr++);
            load_ps1(b_p2_vreg.v, b_p2_pntr++);
            load_ps1(b_p3_vreg.v, b_p3_pntr++); /* load and duplicate */

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

        std::size_t c1 = 1 % left_cols, c2 = 2 % left_cols, c3 = 3 % left_cols;

        for (int i = 0; i < VSIZE; ++i) {
            *(c + i) += ELEMENT(0, 0, 3, i);
            *(c + i + c1 * m) += ELEMENT(0, 1, 3, i);
            *(c + i + c2 * m) += ELEMENT(0, 2, 3, i);
            *(c + i + c3 * m) += ELEMENT(0, 3, 3, i);
            *(c + i + VSIZE) += ELEMENT(4, 0, 7, i);
            *(c + i + VSIZE + c1 * m) += ELEMENT(4, 1, 7, i);
            *(c + i + VSIZE + c2 * m) += ELEMENT(4, 2, 7, i);
            *(c + i + VSIZE + c3 * m) += ELEMENT(4, 3, 7, i);
        }
    }

    template <typename T>
    void PackMatrixA(T *A, int blockrows, int m, int block_k, T *packA, int pad_rows) {
        int i = 0, j = 0;
        for (j = 0; j < block_k; ++j) {
            T *ptr = A + j * m;
            for (i = 0; i < blockrows - pad_rows; ++i) {
                *(packA + i + j * blockrows) = *(ptr + i);
            }
            for (int k = 0; k < pad_rows; ++k) {
                *(packA + i + k + j * blockrows) = 0;
            }
        }
    }

    template <typename T>
    inline void PackMatrixB(T *B, int blockcols, int block_k, int k, T *packB) {
        for (int j = 0; j < blockcols; ++j) {
            for (int i = 0; i < block_k; ++i) {
                *(packB + i + j * block_k) = *(B + i + j * k);
            }
        }
    }

    template <typename T>
    inline void PackMatrixB_pad(int blockcols, int block_k, T *packB) {
        for (int j = 0; j < blockcols; ++j) {
            for (int i = 0; i < block_k; ++i) {
                *(packB + i + j * block_k) = 0;
            }
        }
    }

    template <typename T>
    inline void PackMatrixA_final(T *A, T *packedA, int InnerKernel_rows, int m, int rk, int rm, int pad_rows, int after_pad) {
        int EndVec = after_pad - after_pad % InnerKernel_rows;
        constexpr int half_step = basic_step / (sizeof(T) / sizeof(float));
#pragma omp parallel
        {
#pragma omp for nowait
            for (int i = 0; i < EndVec; i += InnerKernel_rows) {
                if (pad_rows && rm - i < InnerKernel_rows)
                    PackMatrixA(A + i, InnerKernel_rows, m, rk, packedA + rk * i, pad_rows);
                else
                    PackMatrixA(A + i, InnerKernel_rows, m, rk, packedA + rk * i, 0);
            }
            for (int i = EndVec; i < rm; i += half_step) {
                if (pad_rows && rm - i < half_step)
                    PackMatrixA(A + i, half_step, m, rk, packedA + rk * i, half_step - rm + i);
                else
                    PackMatrixA(A + i, half_step, m, rk, packedA + rk * i, 0);
            }
        }
    }

    template <typename T>
    inline void PackMatrixB_final(T *B, T *packedB, int InnerKernel_cols, int n, int rk, int k, int pad_cols) {
#pragma omp parallel
        {
#pragma omp for nowait
            for (int j = 0; j < n; j += InnerKernel_cols)
                PackMatrixB(B + j * k, InnerKernel_cols, rk, k, packedB + j * rk);
            if (pad_cols)
                PackMatrixB_pad(pad_cols, rk, packedB + n * rk);
        }
    }

    template <typename T>
    void InnerKernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int m, int n, int k, bool isFirst, bool isLast) {
        constexpr int InnerKernel_cols = inner_cols;
        constexpr int InnerKernel_rows = 4 * inner_rows / sizeof(T);
        constexpr int half_step = basic_step / (sizeof(T) / sizeof(float)) * 2;

        //alloc memory for packing
        /*std::size_t packedARows = rm % 4 == 0 ? rm : (rm + 4 - rm % 4);
        std::size_t packedBCols = n % 4 == 0 ? n : (n + 4 - n % 4);*/
        std::size_t packedARows = rm % 8 == 0 ? rm : (rm + 8 - rm % 8);
        std::size_t packedBCols = n % 4 == 0 ? n : (n + 4 - n % 4);
        static T *packedA = mynew<T>(packedARows * rk, 16);
        static T *packedB = mynew<T>(packedBCols * rk, 16);
        int EndVecRows = packedARows - packedARows % InnerKernel_rows;

        //pack InnerKernel A(rm*rk) into packedA
        PackMatrixA_final(b_A, packedA, InnerKernel_rows, m, rk, rm, packedARows - rm, packedARows);
        if (isFirst)
            PackMatrixB_final(b_B, packedB, InnerKernel_cols, n, rk, k, packedBCols - n);

#pragma omp parallel for
        for (int j = 0; j < packedBCols; j += InnerKernel_cols) {
            for (int i = 0; i < EndVecRows; i += InnerKernel_rows) {
                AddDot8x4<T, v_256<T>>(packedA + rk * i, packedB + j * rk, C + i + j * m, rk, m, n, k, j);
            }
            for (int i = EndVecRows; i < rm; i += half_step)
                AddDot4x4<T, v_256<T>>(packedA + rk * i, packedB + j * rk, C + i + j * m, rk, m, n, k, j);
        }

        if (isLast) {
            aligned_free(packedB);
            aligned_free(packedA);
        }
    }

    template <typename T>
    void MatDotVectorKernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int m, int n, int k, bool isFirst, bool isLast) {
        const constexpr int InnerKernel_rows = 4 * inner_rows / sizeof(T);

        std::size_t packedARows = rm % 4 == 0 ? rm : (rm + 4 - rm % 4);
        static T *packedA = mynew<T>(packedARows * rk, 16);
        int EndVecRows = rm - rm % InnerKernel_rows;

        PackMatrixA_final(b_A, packedA, InnerKernel_rows, m, rk, rm, packedARows - rm, packedARows);

#pragma omp parallel for
        for (int i = 0; i < EndVecRows; i += InnerKernel_rows) {
            AddDot8x1<T, v_256<T>>(packedA + rk * i, b_B, C + i, rk, m, 1, k);
        }
        for (int i = EndVecRows; i < rm; i += basic_step) {
            AddDot4x1<T, v_128<T>>(packedA + rk * i, b_B, C + i, rk, m, 1, k);
        }

        if (isLast) {
            aligned_free(packedA);
        }
    }

    template <typename lhs, typename rhs, typename dst>
    void Product(lhs *A, rhs *B, dst *C) {
        auto ptrA = A->data();
        auto ptrB = B->data();
        auto ptrC = C->data();
        int p, q, rp = k_kernel, rq = m_kernel;
        int m = A->rows, k = A->cols, n = B->cols;

        if (n != 1) {
            for (p = 0; p < k; p += rp) {
                rp = _min_(k - p, k_kernel);
                for (q = 0; q < m; q += rq) {
                    rq = _min_(m - q, m_kernel);
                    InnerKernel(ptrA + m * p + q, ptrB + p, ptrC + q, rq, rp, n, m, n, k, q == 0, p == k - rp && q == m - rq);
                }
            }
        } else {
            for (p = 0; p < k; p += rp) {
                rp = _min_(k - p, k_kernel);
                for (q = 0; q < m; q += rq) {
                    rq = _min_(m - q, m_kernel);
                    MatDotVectorKernel(ptrA + m * p + q, ptrB + p, ptrC + q, rq, rp, n, m, n, k, q == 0, p == k - rp && q == m - rq);
                }
            }
        }
    }
}  // namespace DDA