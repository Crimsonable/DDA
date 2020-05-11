#pragma once
#include "MetaTools.h"
#include "Expression.h"
#include "OpRegister.h"
#include "DenseStorage.h"
using std::enable_if;

namespace CSM {
	class AssignBase {
	public:
		template<typename Derived, typename Exp, typename BasicBufferType, ENABLE_IF(IS_BASE_OF(CommonBinaryExpBase,Exp)||IS_BASE_OF(CommonSingleExpBase,Exp))>
		FORCE_INLINE static void Assign(Derived* ptr, Exp&& exp) {
			if (ptr->defaultLazyAssign || exp.defaultLazyAssign) {
				ptr->resize(exp.rows, exp.cols);
				exp.template Eval<Derived, BasicBufferType>(ptr);
			}
			else {
				if constexpr (!IS_BASE_OF(CommonMapBase, Derived)) {
					BasicBufferType buffer;
					buffer.resize(exp.rows, exp.cols);
					exp.template Eval<Derived, BasicBufferType>(&buffer);
					ptr->share(&buffer);
				}
			}
			if (exp.deAllocate)
				exp._clear();
		}
	};
}