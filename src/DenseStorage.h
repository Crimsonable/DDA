#pragma once
#include<algorithm>
#include "forwardDecleration.h"
#include "memory.h"
namespace DDA {
	namespace internal {
		template<typename T, int size>
		struct alignas(16) plain_array {
			T array[size+ VECTORIZATION_SIZE / (8 * sizeof(T)) - size % (sizeof(T))];
		};

		template<typename T>
		struct alignas(16) plain_array<T,-1> {
			T* array;
			plain_array(std::size_t size) {
				size += VECTORIZATION_SIZE / (8 * sizeof(T)) - size % (sizeof(T));
				array = reinterpret_cast<T*>(aligned_alloc(size * sizeof(T), 16));
			}

			~plain_array() {
				aligned_free(array);
			}
		};
	}

	template<typename T,int Size,int Rows,int Cols>
	class DenseStorage {
	protected:
		typedef internal::plain_array<T, Size> plainType;
		plainType m_storage;
	public:
		DenseStorage(){}
		DenseStorage(T* _array) {
			memcpy(m_storage.array, _array, Size * sizeof(T));
		}
		DenseStorage(std::initializer_list<T> _l) {
			for (auto&& i = _l.begin(); i < _l.end(); ++i)
                *(m_storage->array + (i - _l.begin())) = *i;
            }
		DenseStorage(const DenseStorage& other) {
            memcpy(m_storage.array, other.cdata(), other.size);
        }
		DenseStorage(const DenseStorage&& other) {
			m_storage = std::move(other.m_storage);
		}
		
		void resize(int) { std::cout << "trying to resize a fixed matrix" << std::endl; }
		void swap(DenseStorage& other) { std::swap(m_storage, other.m_storage); std::swap(rows, other.rows); std::swap(cols, other.cols); }
		inline T* data() { return m_storage.array; }
		inline const T* cdata() const { return m_storage.array; }
		T& operator[](std::size_t idx) { return *(data() + idx); }

		int rows = Rows;
		int cols = Cols;
		int size = Size;
	};

	template<typename T>
	class DenseStorage<T, -1, -1, -1> {
	protected:
		typedef internal::plain_array<T, -1> plainType;
		plainType* m_storage;
	public:
		DenseStorage():cols(-1),rows(-1),size(-1) {}
		~DenseStorage() { delete(m_storage); }
		DenseStorage(T* _array, int _size) {
			if (_size > size) {
                if(size!=-1)
                    delete (m_storage);
				m_storage = new plainType(_size);
				size = _size;
			}
			memcpy(m_storage->array, _array, _size * sizeof(T));
		}

		DenseStorage(std::initializer_list<T> _l) {
			auto listsize = _l.size();
			if (_l.size > size) {
				if(size!=-1)
                    delete (m_storage);
				m_storage = new plainType(listsize);
			}
			size = listsize;
			for (auto&& i = _l.begin(); i < _l.end(); ++i)
				*(m_storage->array + (i - _l.begin())) = *i;
		}

		template<typename otherStorage>
		void operator=(const otherStorage& other) {
			if (other.size > size) {
                if(size!=-1)
                    delete (m_storage);
                m_storage = new plainType(other.size);
				size = other.size;
			}
			memcpy(m_storage->array, other.cdata(), size * sizeof(T));
		}

		void resize(int newSize) {
			if (newSize > size){
                if(size!=-1)
				    delete(m_storage);
				m_storage = new plainType(newSize);
				size = newSize;
			}
		}

		inline T* data() { return m_storage->array; }
		inline const T* cdata() const { return m_storage->array; }
		T& operator[](std::size_t idx) { return *(data() + idx); }

		int rows = -1;
		int cols = -1;
		int size = -1;
	};
}