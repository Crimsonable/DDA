#pragma once
#include "forwardDecleration.h"
#include "Matrix.h"
#include "LuDecompostionImp.h"

namespace CSM {
	namespace Solver {
		template<typename T>
		class LuSolver
		{
			using MatType = typename Matrix<T, -1, -1>;
			using MatType_ptr = std::shared_ptr<MatType>;
			MatType_ptr TempMat;
			bool isSet = false;

		public:
			LuSolver(){}

			template<typename FeedType>
			void set(FeedType&& mat) {
				TempMat = std::make_shared<MatType>();
				TempMat->copy(&mat);
				isSet = true;
			}

			void decompose() {
				//assert(isSet, "No Matrix Feeded!");
				if (isSet)
					LuDecompostion(TempMat->data(), TempMat->rows, TempMat->rows, TempMat->cols);
			}

			void show() {
				TempMat->printMatrix();
			}
		};


	}
}