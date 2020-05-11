#pragma once
#include "forwardDecleration.h"

namespace CSM {
	namespace Solver {
		template<typename T>
		void LuDecompostion(T* A, const int& lda, const int& rows, const int& cols) {
			for (int dIndex = 0; dIndex < rows; ++dIndex) {
				int offset = dIndex * lda;
				T dval = 1.0 / (*(A + dIndex + offset));
				for (int rowIndex = dIndex + 1; rowIndex < rows; ++rowIndex) {
					*(A + offset + rowIndex) = *(A + offset + rowIndex)*dval;
				}
				for (int colIndex = dIndex + 1; colIndex < cols; ++colIndex) {
					int offset = colIndex * lda;
					for (int rowIndex = dIndex + 1; rowIndex < rows; ++rowIndex) {
						*(A + rowIndex + offset) -= *(A + rowIndex + dIndex * lda)**(A + dIndex + offset);
					}
				}
			}
		}

		void LuDecompostion_avx(float* A, const int& lda, const int& rows, const int& cols) {
			__m256 temp;
			for (int dIndex = 0; dIndex < rows; ++dIndex) {
				int offset = dIndex * lda;
				float dval = 1.0 / (*(A + dIndex + offset));
				for (int rowIndex = dIndex + 1; rowIndex < rows; ++rowIndex) {
					*(A + offset + rowIndex) = *(A + offset + rowIndex)*dval;
				}

				for (int colIndex = dIndex + 1; colIndex < cols; ++colIndex) {
					int offset = colIndex * lda;
					for (int rowIndex = dIndex + 1; rowIndex < rows; ++rowIndex) {
						*(A + rowIndex + offset) -= *(A + rowIndex + dIndex * lda)**(A + dIndex + offset);
					}
				}
			}
		}
	}
}