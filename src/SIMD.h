#pragma once
#include "forwardDecleration.h"

namespace DDA {
	template<typename Vtype,typename std::enable_if<
		std::is_same_v<Vtype,__m128>||std::is_same_v<Vtype,__m128d>||std::is_same_v<Vtype,__m256>||std::is_same_v<Vtype,__m256d>,int>::type=0>
	inline Vtype operator*(const Vtype&l, const Vtype&r) {
		if constexpr (std::is_same_v<Vtype, __m128>)
			return _mm_mul_ps(l, r);
		else if constexpr (std::is_same_v<Vtype, __m256>)
			return _mm256_mul_ps(l, r);
		else if constexpr (std::is_same_v<Vtype, __m128d>)
			return _mm_mul_pd(l, r);
		else if constexpr (std::is_same_v<Vtype, __m256d>)
			return _mm256_mul_pd(l, r);
	}

	template<typename Vtype, typename std::enable_if<
		std::is_same_v<Vtype, __m128> || std::is_same_v<Vtype, __m128d> || std::is_same_v<Vtype, __m256> || std::is_same_v<Vtype, __m256d>, int>::type = 0>
		inline Vtype operator+(const Vtype&l, const Vtype&r) {
		if constexpr (std::is_same_v<Vtype, __m128>)
			return _mm_add_ps(l, r);
		else if constexpr (std::is_same_v<Vtype, __m256>)
			return _mm256_add_ps(l, r);
		else if constexpr (std::is_same_v<Vtype, __m128d>)
			return _mm_add_pd(l, r);
		else if constexpr (std::is_same_v<Vtype, __m256d>)
			return _mm256_add_pd(l, r);
	}

	template<typename Vtype,typename T>
	inline void load_ps(Vtype& dst, const T *src) {
		if constexpr (std::is_same_v<Vtype, __m128>)
			dst = _mm_load_ps(src);
		else if constexpr (std::is_same_v<Vtype, __m256>)
			dst = _mm256_load_ps(src);
		else if constexpr (std::is_same_v<Vtype, __m128d>)
			dst = _mm_load_pd(src);
		else if constexpr (std::is_same_v<Vtype, __m256d>)
			dst = _mm256_load_pd(src);
	}

	template<typename Vtype, typename T>
	inline void load_ps1(Vtype& dst, const T *src) {
		if constexpr (std::is_same_v<Vtype, __m128>)
			dst = _mm_load_ps1(src);
		else if constexpr (std::is_same_v<Vtype, __m256>)
			dst = _mm256_broadcast_ss(src);
		else if constexpr (std::is_same_v<Vtype, __m128d>)
			dst = _mm_load1_pd(src);
		else if constexpr (std::is_same_v<Vtype, __m256d>)
			dst = _mm256_broadcast_sd(src);
	}

	template<typename Vtype>
	inline typename internal::traits<Vtype>::vtype set_zeros() {
		using dtype = typename internal::traits<Vtype>::dtype;
		using vtype = typename internal::traits<Vtype>::vtype;
		if constexpr (std::is_same_v<dtype, float>) {
			if constexpr (std::is_same_v<vtype, __m128>)
				return _mm_setzero_ps();
			else if constexpr (std::is_same_v<vtype, __m256>)
				return _mm256_setzero_ps();
		}
		else if constexpr (std::is_same_v<dtype, double>) {
			if constexpr (std::is_same_v<vtype, __m128d>)
				return _mm_setzero_pd();
			else if constexpr (std::is_same_v<vtype, __m256d>)
				return _mm256_setzero_pd();
		}
	}

	template<typename Vtype, typename T>
	inline void store(T *dst, Vtype&& src) {
		if constexpr (std::is_same_v<T, float>) {
			if constexpr (std::is_same_v<Vtype, __m128>)
				_mm_store_ps(dst, src);
			else if constexpr (std::is_same_v<Vtype, __m256>)
				_mm256_store_ps(dst, src);
		}
		else if constexpr (std::is_same_v<T, double>) {
			if constexpr (std::is_same_v<Vtype, __m128d>)
				_mm_store_pd(dst, src);
			else if constexpr (std::is_same_v<Vtype, __m256d>)
				_mm256_store_pd(dst, src);
		}
	}

	template<typename T>
	union v_128
	{
		typename internal::traits<v_128<T>>::vtype v = set_zeros<v_128<T>>();
		T d[128/(8*sizeof(T))];
	};

	template<typename T>
	union v_256 {
		typename internal::traits<v_256<T>>::vtype v = set_zeros<v_256<T>>();
		T d[256/(8*sizeof(T))];
	};

	namespace internal {
		template<>
		struct traits<v_128<float>> {
			using vtype = __m128;
			using dtype = float;
		};

		template<>
		struct traits<v_256<float>> {
			using vtype = __m256;
			using dtype = float;
		};

		template<>
		struct traits<v_128<double>> {
			using vtype = __m128d;
			using dtype = double;
		};

		template<>
		struct traits<v_256<double>> {
			using vtype = __m256d;
			using dtype = double;
		};
	}
}