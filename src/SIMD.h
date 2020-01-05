#pragma once
#include "forwardDecleration.h"

namespace DDA {
	inline __m128 operator*(const __m128& l, const __m128& r) {
		return _mm_mul_ps(l, r);
	}

	inline __m128 operator+(const __m128& l, const __m128& r) {
		return _mm_add_ps(l, r);
	}

	inline __m256 operator*(const __m256& l, const __m256& r) {
		return _mm256_mul_ps(l, r);
	}

	inline __m256 operator+(const __m256& l, const __m256& r) {
		return _mm256_add_ps(l, r);
	}

	template<typename T>
	inline void load_ps(__m128& dst, const T *src) {
		dst = _mm_load_ps(src);
	}

	template<typename T>
	inline void load_ps(__m256& dst, const T *src) {
		dst = _mm256_load_ps(src);
	}

	template<typename T>
	inline void load_ps1(__m128& dst, const T *src) {
		dst = _mm_load_ps1(src);
	}

	template<typename T>
	inline void load_ps1(__m256& dst, const T *src) {
		dst = _mm256_broadcast_ss(src);
	}

	template<typename T>
	union v_128
	{
		__m128 v = _mm_setzero_ps();
		T d[128/(8*sizeof(T))];
	};

	template<typename T>
	union v_256 {
		__m256 v = _mm256_setzero_ps();
		T d[256/(8*sizeof(T))];
	};
}