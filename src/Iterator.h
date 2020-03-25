#pragma once
#include "forwardDecleration.h"
#include <assert.h>

using std::enable_if_t;
using std::is_arithmetic_v;

namespace DDA {
	template<typename DerivedIterator>
	class IteratorBase {
	private:
		DerivedIterator *ptr = derived();

		inline DerivedIterator* derived() {
			return static_cast<DerivedIterator*>(this);
		}
	};

	template<typename MatType>
	class Iterator :public IteratorBase<Iterator<MatType>> {
	private:
		using scalar = typename internal::traits<MatType>::scalar;
		MatType	*mat_ptr;
		scalar *data_ptr;
		unsigned int counter = 0;
	public:
		Iterator(MatType& mat):mat_ptr(&mat),data_ptr(mat.data()){}

		inline scalar& operator* () const {
			return *(data_ptr + counter);
		}

		inline void operator++ () {
			assert(counter <= mat_ptr->size);
			counter++;
		}

		inline void operator-- () {
			assert(counter - 1 >= 0);
			counter--;
		}

		inline void begin() {
			counter = 0;
		}

		inline void end() {
			counter = mat_ptr->size;
		}

		template<typename T,enable_if_t<is_arithmetic_v<T>,void>>
		void operator= (const T& other) {
			*(data_ptr + counter) = other;
		}
		
		template<typename OtherIterator>
		inline bool operator== (const OtherIterator& it) {
			return it.mat_ptr == mat_ptr && it.counter == counter;
		}

		template<typename OtherIterator>
		inline bool operator!= (const OtherIterator& it) {
			return it.mat_ptr == mat_ptr && it.counter != counter;
		}
	};
}