#pragma once
#include "forwardDecleration.h"
#include "Transpose.h"

namespace DDA {
	template<typename T>
	FORCE_INLINE void pack_8x8(T *src, int ld, int rows, int cols, T *dst, int ld2) {
#pragma unroll
		for (int colIndex = 0; colIndex < 8; ++colIndex) {
			int offset = colIndex * ld2;
			if (colIndex >= cols) {
				std::fill_n(dst + offset, 8, T(0));
			}
			else {
				memcpy(dst + offset, src + ld * colIndex, sizeof(T)*rows);
				std::fill_n(dst + offset + rows, 8 - rows, T(0));
			}
		}
	}

	FORCE_INLINE void pack_8x8_transpose(float *src, int ld, int rows, int cols, float *dst, int ld2) {
		/*float temp[64];
		pack_8x8(src, ld, rows, cols, temp, 8);
		tran(temp, 8, dst, ld2);*/
		for (int colIndex = 0; colIndex < cols; ++colIndex) {
			for (int rowIndex = 0; rowIndex < rows; ++rowIndex) {
				*(dst + colIndex + rowIndex * ld2) = *(src + rowIndex + colIndex * ld);
			}
		}
	}
}