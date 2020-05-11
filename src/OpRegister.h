#pragma once
namespace CSM {
	namespace internal {
		enum OpRegister
		{
			None,
			MatMul,
			MatAdd,
			MatSub,
			CwiseMul,
			Transpose
		};
	}
}