#pragma once
#include<algorithm>
namespace DDA {
	namespace internal {
		template<class T, int size>
		struct plain_array {
			T array[size];
			plain_array(){}
		};
	}

	template<class T, int size, int Rows, int Cols> class DenseStroage;

	template<class T,int size,int Rows,int Cols>
	class DenseStorage {
		internal::plain_array<T, size> m_storage;
		int rows, cols;
	public:
		DenseStorage(){}
		DenseStorage(const DenseStorage& other):m_storage(other.m_storage),rows(Rows),cols(Cols){}
		DenseStorage(const DenseStorage&& other) :m_storage(std::move(other.m_storage)), rows(other.rows), cols(other.cols) {}

		void swap(DenseStorage& other) { std::swap(m_storage, other.m_storage); std::swap(rows, other.rows); std::swap(cols, other.cols); }
		T* data() { return m_storage.array; }
	};
}