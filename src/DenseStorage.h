#pragma once
#include <algorithm>
#include "forwardDecleration.h"
#include "memory.h"
namespace CSM
{
	namespace Memory {
		template <typename T, int size>
		struct alignas(VECTORIZATION_ALIGN_BYTES) plain_array
		{
			T array[size + 64];
		};

		template <typename T>
		struct alignas(VECTORIZATION_ALIGN_BYTES) plain_array<T, -1>
		{
			T *array;
			plain_array(std::size_t size)
			{
				size += 64;
				array = mynew_fill0<T>(size, VECTORIZATION_ALIGN_BYTES);
			}

			~plain_array()
			{
				aligned_free(array);
			}
		};
	}

	template<typename Derived>
	class DenseStorageBase {
	public:
		FORCE_INLINE Derived* derived() {
			return static_cast<Derived*>(this);
		}
	};

	template<typename T>
	class DenseStorage :public DenseStorageBase<DenseStorage<T>> {
	private:
		using plainType = typename Memory::plain_array<T, -1>;
		using plainType_ptr = typename std::shared_ptr<plainType>;
		plainType_ptr m_storage;

	public:
		DenseStorage(){}
		DenseStorage(int rows, int cols) {
			resize(rows, cols);
		}

		void resize(int _size) {
			if (_size > capacity) {
				m_storage = std::make_shared<plainType>(_size);
				capacity = _size;
			}
			size = _size;
		}

		void resize(int _rows, int _cols) {
			int newSize = _rows * _cols;
			if (newSize > capacity) {
				m_storage = std::make_shared<plainType>(newSize);
				capacity = newSize;
			}
			rows = _rows;
			ld = rows;
			cols = _cols;
			size = newSize;
		}

		template<typename Container>
		void copy(Container* src) {
			resize(src->rows, src->cols);
			memcpy(data(), src->data(), sizeof(T)*rows*cols);
		}
		
		template <typename Container>
		inline void share(Container* other_storage) {
			if (other_storage->m_storage == m_storage) return;
			m_storage = other_storage->m_storage;
			capacity = other_storage->capacity;
			size = other_storage->size;
			rows = other_storage->rows;
			cols = other_storage->cols;
			ld = rows;
		}

		template <typename Container>
		inline void swap(Container* other_storage) {
			if (other_storage->m_storage == m_storage) return;
			size = other_storage->size;
			rows = other_storage->rows;
			ld = rows;
			cols = other_storage->cols;
			capacity = other_storage->capacity;
			std::swap(m_storage, other_storage->m_storage);
		}

		FORCE_INLINE T *data() { return m_storage.get()->array; }
		inline const T *cdata() const { return m_storage->array; }

		int rows = -1, cols = -1, size = -1, capacity = -1, ld = -1;
	};

	template<typename T>
	class DenseStorageMap :public DenseStorageBase<DenseStorageMap<T>> {
	private:
		T* m_storage;
	public:
		DenseStorageMap(){}
		DenseStorageMap(T* data, int ld, int rows, int cols):m_storage(data),ld(ld),rows(rows),cols(cols),size(rows*cols) {}

		void resize(int _rows, int _cols) {
			if (rows != _rows || cols != _cols)
				abort();
		}

		template<typename Container>
		void share(Container* otherStorage){}

		template<typename Container>
		void swap(Container* otherStorage) {}

		FORCE_INLINE T* data() { return m_storage; }
		FORCE_INLINE const T* cdata() { return m_storage; }

		int rows = -1, cols = -1, size = -1, capacity = -1, ld = -1;
	};
}

	/*template <typename T, int Size, int Rows, int Cols>
	class DenseStorage
	{
	protected:
		using plainType = internal::plain_array<T, Size>;
		plainType m_storage;

	public:
		DenseStorage() {}
		DenseStorage(T *_array)
		{
			memcpy(m_storage.array, _array, Size * sizeof(T));
		}
		DenseStorage(std::initializer_list<T> _l)
		{
			for (auto &&i = _l.begin(); i < _l.end(); ++i)
				*(m_storage.array + (i - _l.begin())) = *i;
		}
		DenseStorage(const DenseStorage &other)
		{
			DenseStorage(other.data());
		}
		DenseStorage(const DenseStorage &&other)
		{
			m_storage = std::move(other.m_storage);
		}

		void resize(int, int) {}
		//void swap(DenseStorage& other) { std::swap(m_storage, other.m_storage); std::swap(rows, other.rows); std::swap(cols, other.cols); }
		inline T *data() { return m_storage.array; }
		inline const T *cdata() const { return m_storage.array; }
		T &operator[](std::size_t idx) { return *(data() + idx); }

		int rows = Rows;
		int cols = Cols;
		int size = Size;
	};

	template <typename T>
	class DenseStorage<T, -1, -1, -1>
	{
	protected:
		using plainType = typename internal::plain_array<T, -1>;
		using plainType_ptr = typename std::shared_ptr<plainType>;
		plainType_ptr m_storage;
		bool isPad = false;

	public:
		DenseStorage() : cols(-1), rows(-1), size(-1) {}
		~DenseStorage()
		{
			//delete(m_storage);
		}

		void operator=(const DenseStorage &other)
		{
			resize(other.rows, other.cols);
			share(const_cast<DenseStorage *>(&other));
			cols = other.cols;
			rows = other.rows;
		}

		void resize(int newRows, int newCols)
		{
			int newSize = newRows * newCols;
			if (newSize > capacity)
			{
				m_storage = std::make_shared<plainType>(newSize);
				capacity = newSize;
				ld = newRows;
				width = newCols;
			}
			size = newSize;
			cols = newCols;
			rows = newRows;
		}

		void resize(int _size) {
			if (_size > capacity) {
				m_storage = std::make_shared<plainType>(_size);
				capacity = _size;
			}
			size = _size;
		}

		template <typename otherDenseStorage>
		inline void share(otherDenseStorage other_storage)		
		//other_storage must be a pointer
		{
			//m_storage.reset(); may cause UB		
			if (other_storage->m_storage == m_storage) return;
			m_storage = other_storage->m_storage;
			capacity = other_storage->capacity;
			size = other_storage->size;
			rows = other_storage->rows;
			cols = other_storage->cols;
			ld = other_storage->ld;
			width = other_storage->width;
		}

		template <typename otherDenseStorage>
		inline void swap(otherDenseStorage other_storage)		
		//other_storage must be a pointer
		{
			if (other_storage->m_storage == m_storage) return;
			size = other_storage->size;
			rows = other_storage->rows;
			cols = other_storage->cols;
			ld = other_storage->ld;
			width = other_storage->width;
			capacity = other_storage->capacity;
			std::swap(m_storage, other_storage->m_storage);
		}

		template<typename DenseStorage_ptr>
		inline void copy(DenseStorage_ptr ptr) {
			resize(ptr->size);
			rows = ptr->rows;
			cols = ptr->cols;
			size = ptr->size;
			for(int colIndex=0;colIndex<)
		}

		FORCE_INLINE T *data() { return m_storage.get()->array; }
		inline const T *cdata() const { return m_storage->array; }
		T &operator[](std::size_t idx) { return *(data() + idx); }

		int rows = -1;
		int cols = -1;
		int ld = -1;
		int width = -1;
		int size = -1;
		int capacity = -1;
	};

	template<typename T>
	class DenseStorageForMap{

	}

} // namespace CSM*/