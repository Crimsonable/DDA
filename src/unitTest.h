#pragma once
#include <chrono>
#include <string>
#include <vector>
#include "forwardDecleration.h"
#include "Rua.h"
#include "eigen/Eigen/Dense"

namespace CSM {
    template <typename dtype>
    class Test {
		using myClock = typename std::chrono::high_resolution_clock::time_point;
        using myTimeSpan = typename std::chrono::duration<double>;
        using TestFuncRetType = typename std::tuple<bool, double, double, double, double>;
        using FuncType = typename std::function<TestFuncRetType(const int, const int, const int, int)>;
		using LoopFuncRetType = std::tuple<double, double, double, double>;
		using MatType = Matrix<dtype, -1, -1>;
		using Eigen_MatType = Eigen::Matrix<dtype, -1, -1>;
		using MatType_ptr = std::shared_ptr<MatType>;
		using Eigen_MatType_ptr = std::shared_ptr<Eigen_MatType>;

    public:
		Test(const int M, const int K, const int N, int numThread) :M(M), N(N), K(K), numThread(numThread) {
#ifndef _DEBUG
			omp_set_num_threads(numThread);
#endif // !_DEBUG
			a = std::make_shared<MatType>();
			a->resize(M, K);
			a->setRandom();
			b = std::make_shared<MatType>();
			b->resize(K, N);
			b->setRandom();
			c = std::make_shared<MatType>();
			c->resize(M, N);
#ifdef EIGEN_BENCHMARK
			ea = std::make_shared<Eigen_MatType>();
			eb = std::make_shared<Eigen_MatType>();
			ec = std::make_shared<Eigen_MatType>();	
			ea->resize(M,K);
			eb->resize(K, N);
			ec->resize(M, N);
			ec->setZero();
			for (int i = 0; i < M * K; ++i) {
				ea->coeffRef(i) = a->coeff(i);
			}
			for (int i = 0; i < K * N; ++i) {
				eb->coeffRef(i) = b->coeff(i);
			}
#endif // EIGEN_BENCHMARK
		}

		template<typename ...Argc>
		void BenchImp(Argc&&... args) {
			t0 = std::chrono::high_resolution_clock::now();
			//ExpressionFunctionTestGemm(std::forward<Argc>(args)...);
			ProductFunctionTestGemm(std::forward<Argc>(args)...);
			t1 = std::chrono::high_resolution_clock::now();
			timeSpan0= std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
		}

		template<typename Matptr>
		FORCE_INLINE void ProductFunctionTest(Matptr lhs, Matptr rhs, Matptr dst) {
			if constexpr (std::is_base_of_v<CommonBase, typename Matptr::element_type>)
				Product(lhs->data(), rhs->data(), dst->data(), lhs->rows, rhs->rows, lhs->rows, rhs->cols, rhs->rows);
			else
				*dst = *lhs**rhs;
		}

		template<typename Matptr>
		FORCE_INLINE void ProductFunctionTestGemm(Matptr lhs, Matptr rhs, Matptr dst) {
			if constexpr (std::is_base_of_v<CommonBase, typename Matptr::element_type>)
				Gemm(lhs->data(), M, rhs->data(), K, dst->data(), M, M, N, K);
			else
				*dst = *lhs**rhs;
		}

		template<typename Matptr>
		FORCE_INLINE void ExpressionFunctionTestGemm(Matptr lhs, Matptr rhs, Matptr dst) {
			*dst = *lhs**rhs;
		}

		void Bench(int times) {
			double min_count_Exp = std::numeric_limits<double>::max();
			double min_count_Eigen = std::numeric_limits<double>::max();
			double ExpCount = 0.0, EigenCount = 0.0;
			for (int i = 0; i < times; ++i) {
				BenchImp(a, b, c);
				min_count_Exp = min_count_Exp > timeSpan0.count() ? timeSpan0.count() : min_count_Exp;
				ExpCount += timeSpan0.count();
#ifdef EIGEN_BENCHMARK
				BenchImp(ea, eb, ec);
				min_count_Eigen = min_count_Eigen > timeSpan0.count() ? timeSpan0.count() : min_count_Eigen;
				EigenCount += timeSpan0.count();
				if (i == 0) {
					double ecsum = ec->sum();
					double csum = c->sum();
					if (abs(csum - ecsum) / ecsum > 1e-5) {
						cout << "Wrong Answer!" << endl;
						cout << "Eigen Answer: " << ecsum << endl;
						cout << "Exp Answer: " << csum << endl;
						cout << "Exp: " << endl;
						c->printMatrix();
						cout << "Eigen: " << endl;
						cout << *ec;
						return;
					}
				}
#endif // EIGEN_BENCHMARK
			}
			double Gflops_Exp = 1e-9 * 2 * M*N*K / ExpCount * times;
			double Gflops_Exp_Max = 1e-9 * 2 * M*N*K / min_count_Exp;
			cout << "Exp: " << endl;
			cout << "Average Gflops: " << Gflops_Exp << endl;
			cout << "Max Gflops: " << Gflops_Exp_Max << endl;
#ifdef EIGEN_BENCHMARK
			double Gflops_Eigen_Max = 1e-9 * 2 * M*N*K / min_count_Eigen;
			double Gflops_Eigen = 1e-9 * 2 * M*N*K / EigenCount * times;
			cout << "Eigen: " << endl;
			cout << "Average Gflops: " << Gflops_Eigen << endl;
			cout << "Max Gflops: " << Gflops_Eigen_Max << endl;
#endif // EIGEN_BENCHMARK
		}

		myClock t0, t1;
		myTimeSpan timeSpan0 = std::chrono::duration<double>();
		MatType_ptr a, b, c;
		Eigen_MatType_ptr ea, eb, ec;
		int M, N, K, numThread;
    };

	template<typename Func,typename ...Args>
	void Bench(int times, Func&& f, Args&&... args) {
		auto t0 = std::chrono::high_resolution_clock::now();
		for (int t = 0; t < times; ++t)
			f(std::forward<Args>(args)...);
		auto t1 = std::chrono::high_resolution_clock::now();
		auto span= std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
		std::cout << "Time Cost: " << span.count()/times << endl;
	}
}