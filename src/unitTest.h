#pragma once
#include <chrono>
#include"forwardDecleration.h"
#include "eigen/Eigen/Dense"
#include <string>
#include <vector>
#include "Matrix.h"
#include "MatrixXpr.h"
#include "Product.h"


namespace DDA {
    template <typename dtype>
    class Test {
        using myClock = typename std::chrono::steady_clock::time_point;
        using myTimeSpan = typename std::chrono::duration<double>;
		using TestFuncRetType = typename std::tuple<bool, double, double, double, double>;
		using FuncType = typename std::function<TestFuncRetType(const int, const int, const int, int)>;
		
		myClock t0, t1, t2, t3;
		myTimeSpan timeSpan0 = std::chrono::duration<double>(), timeSpan1 = std::chrono::duration<double>();

    public:
		TestFuncRetType TestForMatDotPerforemence(const int M, const int K, const int N, int numThread) {
#ifndef _DEBUG
            omp_set_num_threads(numThread);
#endif
            Matrix<dtype, -1, -1> a, b, c;
            a.resize(M, K);
            a.setRandom();
            b.resize(K, N);
            b.setRandom();
            c.resize(M, N);
            c.setZeros();
            t0 = std::chrono::steady_clock::now();
            Product(&a, &b, &c);
            t1 = std::chrono::steady_clock::now();
            timeSpan0 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
            
#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(c.data(), M, N);
#endif // DEBUG_INFO
			
#ifdef EIGEN_BENCHMARK
            Eigen::Matrix<dtype, -1, -1> ea, eb, ec;
            ea.resize(M, K);
            eb.resize(K, N);
            ec.resize(M, N);
            ec.setZero();
            for (int i = 0; i < M * K; ++i) {
                ea.coeffRef(i) = a.coeff(i);
            }
            for (int i = 0; i < K * N; ++i) {
                eb.coeffRef(i) = b.coeff(i);
            }
            t2 = std::chrono::steady_clock::now();
            ec = ea * eb;
            t3 = std::chrono::steady_clock::now();
            timeSpan1 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
#ifdef DEBUG_INFO
			std::cout << ec;
#endif // DEBUG_INFO
			return { true,timeSpan0.count(),timeSpan1.count() ,c.sum(),ec.sum() };
#endif

#ifndef EIGEN_BENCHMARK
			return { false,timeSpan0.count(),0,c.sum(),0 };
#endif // !EIGEN_BENCHMARK
        }

		void Loop(int n, const int M, const int K, const int N, int threads) {
			double total_t = 0, total_et = 0, sum, sume;
			bool flag;
			for (int i = 0; i < n; ++i) {
				double t, et;
				auto counter = TestForMatDotPerforemence(M, K, N, threads);
				std::tie(flag, t, et, sum, sume) = counter;
#ifdef EIGEN_BENCHMARK
				if (!check_by_sum(sum, sume)) {
					std::cout << "Wrong answer!" << std::endl;
					std::cout << "Eigen Sum: " << sume << std::endl;
					std::cout << "Exp Sum: " << sum << std::endl;
					return;
			}
#endif // EIGEN_BENCHMARK
				if (flag)
					total_et += et;
				total_t += t;
			}
			printTestRes("Matrix Dot Test :" + std::to_string(M) + '*' + std::to_string(K) + '*' + std::to_string(N),
				double(total_t / n), double(2 * 1e-9 * M*N*K / (total_t / n)));
			if (flag) {
				printTestRes("Eigen Matrix Dot :" + std::to_string(M) + '*' + std::to_string(K) + '*' + std::to_string(N),
					double(total_et / n), double(2 * 1e-9 * M*N*K / (total_et / n)));
				check_by_sum(sum, sume, true);
			}
		}

        bool check_by_sum(const double& c, const double& ec, bool flag=false) {
			if (flag) {
				std::cout << "Exp Sum: " << c << std::endl;
				std::cout << "Eigen Sum: " << ec << std::endl;
			}
			return abs(ec - c) / ec < 1e-4;
        }

        void printTestRes(const std::string& s, const double& span, double Gfloaps) {
            std::cout << s << std::endl;
            std::cout << "cost: " << span << std::endl;
            std::cout << "Gfloaps: " << Gfloaps << std::endl;
        }
    };
}  // namespace DDA