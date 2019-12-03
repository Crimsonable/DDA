#pragma once
#include<algorithm>
#include "forwardDecleration.h"
namespace DDA {
	namespace internal {
		template<typename T, int size>
		struct __attribute__ ((aligned (16))) plain_array {
			//std::unique_ptr<T[]> array;
			T array[size+ VECTORIZATION_SIZE / (8 * sizeof(T)) - size % (sizeof(T))];
			/*plain_array(){
				//array = std::make_unique<T[]>(size);
			}*/
		};
	}

	template<typename T,int Size,int Rows,int Cols>
	class DenseStorage {
	protected:
		typedef internal::plain_array<T, Size> plainType;
		//std::unique_ptr<plainType> m_storage;
		plainType m_storage;
	public:
		DenseStorage(){
			//m_storage = std::make_unique<plainType>();
		}
		DenseStorage(T* _array) {
			/*if(!m_storage)
				m_storage = std::make_unique<plainType>();*/
			memcpy(m_storage.array, _array, Size * sizeof(T));
		}
		DenseStorage(std::initializer_list<T> _l) {
			//m_storage = std::make_unique<plainType>();
			for (auto&& i = _l.begin(); i < _l.end(); ++i)
				*(m_storage->array + (i - _l.begin())) = *i;
		}
		DenseStorage(const DenseStorage& other) {
			DenseStorage(other.data());
		}
		DenseStorage(const DenseStorage&& other) {
			//m_storage.reset();
			m_storage = std::move(other.m_storage);
		}
		
		void resize(std::size_t) { std::cout << "trying to resize a fixed matrix" << std::endl; }
		void swap(DenseStorage& other) { std::swap(m_storage, other.m_storage); std::swap(rows, other.rows); std::swap(cols, other.cols); }
		T* data() { return m_storage.array; }
		const T* cdata() const { return m_storage.array; }
		T& operator[](std::size_t idx) { return *(data() + idx); }

		int rows = Rows;
		int cols = Cols;
		int size = Size;
	};
}