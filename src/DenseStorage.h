#pragma once
#include <algorithm>
#include "forwardDecleration.h"
#include "memory.h"
namespace DDA
{
	namespace internal
	{
		static enum PadType
		{
			NoPad,
			PadWith4,
			PadWith8,
			PadWith16
		};

		template <typename T, int size>
		struct alignas(VECTORIZATION_ALIGN_BYTES) plain_array
		{
			T array[size + 8];
		};

		template <typename T>
		struct alignas(VECTORIZATION_ALIGN_BYTES) plain_array<T, -1>
		{
			T *array;
			plain_array(std::size_t size)
			{
				size += 8;
				array = mynew_fill0<T>(size, VECTORIZATION_ALIGN_BYTES);
			}

			~plain_array()
			{
				aligned_free(array);
			}
		};
	} // namespace internal

	template <typename T, int Size, int Rows, int Cols>
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
		DenseStorage(T *_array, int _size)
		{
			if (_size > size)
			{
				//m_storage = new plainType(_size);
				m_storage = std::make_shared<plainType>(_size);
				size = _size;
			}
			memcpy(m_storage->array, _array, _size * sizeof(T));
		}

		DenseStorage(std::initializer_list<T> _l)
		{
			auto listsize = _l.size();
			if (_l.size > size)
			{
				m_storage.reset(new plainType(listsize));
			}
			size = listsize;
			for (auto &&i = _l.begin(); i < _l.end(); ++i)
				*(m_storage->array + (i - _l.begin())) = *i;
		}

		void operator=(const DenseStorage &other)
		{
			resize(other.rows, other.cols);
			//memcpy(m_storage->array, other.cdata(), size * sizeof(T));
			share(const_cast<DenseStorage *>(&other));
			cols = other.cols;
			rows = other.rows;
		}

		void resize(int newRows, int newCols)
		{
			int newSize = newRows * newCols;
			if (newSize > size)
			{
				m_storage = std::make_shared<plainType>(newSize);
			}
			size = newSize;
			cols = newCols;
			rows = newRows;
		}

		template <typename otherDenseStorage>
		inline void share(otherDenseStorage other_storage)		
		//other_storage must be a pointer
		{
			//m_storage.reset(); may cause UB								
			m_storage = other_storage->m_storage;
			size = other_storage->size;
			rows = other_storage->rows;
			cols = other_storage->cols;
		}

		template <typename otherDenseStorage>
		inline void swap(otherDenseStorage other_storage)		
		//other_storage must be a pointer
		{
			std::swap(m_storage, other_storage->m_storage);
		}

		inline T *data() { return m_storage.get()->array; }
		inline const T *cdata() const { return m_storage->array; }
		T &operator[](std::size_t idx) { return *(data() + idx); }

		int rows = -1;
		int cols = -1;
		int size = -1;
	};

} // namespace DDA