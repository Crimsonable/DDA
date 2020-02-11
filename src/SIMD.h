#pragma once
#include "forwardDecleration.h"

namespace DDA {
	namespace SSE_OP {
		using DDA::internal::traits;
		template<typename Vtype, typename std::enable_if<
			std::is_same_v<Vtype, __m128> || std::is_same_v<Vtype, __m128d> || std::is_same_v<Vtype, __m256> || std::is_same_v<Vtype, __m256d>, int>::type = 0>
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

		template<typename Vtype, typename std::enable_if<
			std::is_same_v<Vtype, __m128> || std::is_same_v<Vtype, __m128d> || std::is_same_v<Vtype, __m256> || std::is_same_v<Vtype, __m256d>, int>::type = 0>
			inline Vtype fmadd(const Vtype&a, const Vtype&b, const Vtype&c) {
			if constexpr (std::is_same_v<Vtype, __m128>)
				return _mm_fmadd_ps(a, b, c);
			else if constexpr (std::is_same_v<Vtype, __m256>)
				return _mm256_fmadd_ps(a, b, c);
			else if constexpr (std::is_same_v<Vtype, __m128d>)
				return _mm_fmaddsub_pd(a, b, c);
			else if constexpr (std::is_same_v<Vtype, __m256d>)
				return _mm256_fmadd_pd(a, b, c);
		}


		template<typename Vtype, typename T>
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

		template<typename Vtype, typename Dtype>
		inline Vtype set_zeros() {
			if constexpr (std::is_same_v<Dtype, float>) {
				if constexpr (std::is_same_v<Vtype, __m128>)
					return _mm_setzero_ps();
				else if constexpr (std::is_same_v<Vtype, __m256>)
					return _mm256_setzero_ps();
			}
			else if constexpr (std::is_same_v<Dtype, double>) {
				if constexpr (std::is_same_v<Vtype, __m128d>)
					return _mm_setzero_pd();
				else if constexpr (std::is_same_v<Vtype, __m256d>)
					return _mm256_setzero_pd();
			}
			else if constexpr (std::is_same_v<Dtype, int>) {
				if constexpr (std::is_same_v<Vtype, __m128i>)
					return _mm_setzero_si128();
				else if constexpr (std::is_same_v<Vtype, __m256i>)
					return _mm256_setzero_si256();
			}
		}

		template<typename Vtype, typename T>
		inline void store(T *dst, Vtype&& src) {
			using real_Vtype = std::remove_reference_t<Vtype>;
			if constexpr (std::is_same_v<T, float>) {
				if constexpr (std::is_same_v<real_Vtype, __m128>)
					_mm_store_ps(dst, src);
				else if constexpr (std::is_same_v<real_Vtype, __m256>)
					_mm256_store_ps(dst, src);
			}
			else if constexpr (std::is_same_v<T, double>) {
				if constexpr (std::is_same_v<real_Vtype, __m128d>)
					_mm_store_pd(dst, src);
				else if constexpr (std::is_same_v<real_Vtype, __m256d>)
					_mm256_store_pd(dst, src);
			}
		}

		template<typename Vtype, typename T, typename Mask>
		inline void store_mask(T *dst, Mask& mask, Vtype&& src) {
			using real_Vtype = std::remove_reference_t<Vtype>;
			if constexpr (std::is_same_v<T, float>) {
				if constexpr (std::is_same_v<real_Vtype, __m128>)
					_mm_maskstore_ps(dst, mask, src);
				else if constexpr (std::is_same_v<real_Vtype, __m256>)
					_mm256_maskstore_ps(dst, mask, src);
			}
			else if constexpr (std::is_same_v<T, double>) {
				if constexpr (std::is_same_v<real_Vtype, __m128d>)
					_mm_maskstore_pd(dst, mask, src);
				else if constexpr (std::is_same_v<real_Vtype, __m256d>)
					_mm256_maskstore_pd(dst, mask, src);
			}
		}

		template<typename T>
		union v_128
		{
			typename traits<v_128<T>>::vtype v = set_zeros<typename traits<v_128<T>>::vtype,T>();
			T d[128 / (8 * sizeof(T))];
		};

		template<typename T>
		union v_256 {
			typename traits<v_256<T>>::vtype v = set_zeros<typename traits<v_256<T>>::vtype, T>();
			T d[256 / (8 * sizeof(T))];
		};

		template<typename Vtype>
		union Mask {
			typename traits<Vtype>::mask_type v = set_zeros<typename traits<Vtype>::mask_type,int>();
			typename traits<Vtype>::mask_int d[sizeof(Vtype) / sizeof(typename traits<Vtype>::dtype)];
		};
	}
	namespace internal {
		using namespace SSE_OP;
		template<>
		struct traits<v_128<float>> {
			using vtype = __m128;
			using dtype = float;
			using mask_type = __m128i;
			using mask_int = unsigned int;
		};

		template<>
		struct traits<v_256<float>> {
			using vtype = __m256;
			using dtype = float;
			using mask_type = __m256i;
			using mask_int = unsigned int;
		};

		template<>
		struct traits<v_128<double>> {
			using vtype = __m128d;
			using dtype = double;
			using mask_type = __m128i;
			using mask_int = unsigned long long int;
		};

		template<>
		struct traits<v_256<double>> {
			using vtype = __m256d;
			using dtype = double;
			using mask_type = __m256i;
			using mask_int = unsigned long long int;
		};

		template<>
		struct traits<v_128<int>> {
			using vtype = __m128i;
			using dtype = int;
		};

		template<>
		struct traits<v_256<int>> {
			using vtype = __m256i;
			using dtype = int;
		};
	}
}