#pragma once
#include "forwardDecleration.h"
namespace CSM {
	FORCE_INLINE void* aligned_alloc(std::size_t size, std::size_t alignment) {
		if (alignment & (alignment - 1))
			return nullptr;
		else {
			void* ptr_head = ::operator new(size + alignment + sizeof(void*));
			if (ptr_head) {
				void* p_buff = reinterpret_cast<void*>(reinterpret_cast<std::size_t>(ptr_head) + sizeof(void*));
				void* aligned_ptr = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(p_buff) | (alignment - 1)) + 1);
				*reinterpret_cast<void**>(reinterpret_cast<std::size_t>(aligned_ptr) - sizeof(void*)) = ptr_head;			
				return aligned_ptr;
			}
			else
				return nullptr;
		}
	}

	FORCE_INLINE void aligned_free(void* ptr) {
		std::free(*reinterpret_cast<void**>(reinterpret_cast<std::size_t>(ptr) - sizeof(void*)));
		ptr = nullptr;
	}

	template<typename T>
	FORCE_INLINE T* mynew_fill0(std::size_t size, std::size_t alignment) {
		T *ptr = reinterpret_cast<T*>(aligned_alloc(size * sizeof(T), alignment));
		std::fill_n(ptr, size, T(0));
		return ptr;
	}

	template<typename T>
	FORCE_INLINE T* mynew(std::size_t size, std::size_t alignment) {
		T *ptr = reinterpret_cast<T*>(aligned_alloc(size * sizeof(T), alignment));
		return ptr;
	}
}