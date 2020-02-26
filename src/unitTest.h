#pragma once
#include <chrono>
#include <string>
#include <vector>
#include "Matrix.h"
#include "BinaryOp.h"
#include "Product.h"
#include "eigen/Eigen/Dense"
#include "forwardDecleration.h"
#include "SingleOp.h"

namespace DDA {
    template <typename dtype>
    class Test {
        using myClock = typename std::chrono::steady_clock::time_point;
        using myTimeSpan = typename std::chrono::duration<double>;
        using TestFuncRetType = typename std::tuple<bool, double, double, double, double>;
        using FuncType = typename std::function<TestFuncRetType(const int, const int, const int, int)>;
		using LoopFuncRetType = std::tuple<double, double, double, double>;
		using MatType = Matrix<dtype, -1, -1>;
		using Eigen_MatType = Eigen::Matrix<dtype, -1, -1>;
		using MatType_ptr = std::shared_ptr<MatType>;
		using Eigen_MatType_ptr = std::shared_ptr<Eigen_MatType>;

        myClock t0, t1, t2, t3;
        myTimeSpan timeSpan0 = std::chrono::duration<double>(), timeSpan1 = std::chrono::duration<double>();
		MatType_ptr a, b, c;
		Eigen_MatType_ptr ea, eb, ec;
		int M, N, K, numThread;

    public:
		Test(const int M, const int K, const int N, int numThread) :M(M), N(N), K(K), numThread(numThread) {
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
			ea->resize(M, K);
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

#ifndef _DEBUG
			omp_set_num_threads(numThread);
#endif
		}

        TestFuncRetType TestForMatDotPerforemence() {
			c->setZeros();
            t0 = std::chrono::steady_clock::now();
			Product(a.get(), b.get(), c.get());
            t1 = std::chrono::steady_clock::now();
            timeSpan0 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);

#ifdef DEBUG_INFO
            DEBUG_TOOLS::printRawMatrix(c->data(), M, N);
#endif  // DEBUG_INFO

#ifdef EIGEN_BENCHMARK
            t2 = std::chrono::steady_clock::now();
            *ec = *ea * *eb;
            t3 = std::chrono::steady_clock::now();
            timeSpan1 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
#ifdef DEBUG_INFO
            std::cout << ec;
#endif  // DEBUG_INFO
            return {true, timeSpan0.count(), timeSpan1.count(), c->sum(), ec->sum()};
#endif

#ifndef EIGEN_BENCHMARK
            return {false, timeSpan0.count(), 0, c->sum(), 0};
#endif  // !EIGEN_BENCHMARK
        }

		TestFuncRetType TestForMatDotPerforemence_para(int rk) {
			c->setZeros();
			t0 = std::chrono::steady_clock::now();
			Product(a.get(), b.get(), c.get(), rk);
			t1 = std::chrono::steady_clock::now();
			timeSpan0 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);

#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(c->data(), M, N);
#endif  // DEBUG_INFO

#ifdef EIGEN_BENCHMARK
			t2 = std::chrono::steady_clock::now();
			*ec = *ea * *eb;
			t3 = std::chrono::steady_clock::now();
			timeSpan1 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
#ifdef DEBUG_INFO
			std::cout << ec;
#endif  // DEBUG_INFO
			return { true, timeSpan0.count(), timeSpan1.count(), c->sum(), ec->sum() };
#endif

#ifndef EIGEN_BENCHMARK
			return { false, timeSpan0.count(), 0, c->sum(), 0 };
#endif  // !EIGEN_BENCHMARK
		}

        TestFuncRetType TestForMatExpression() {
            t0 = std::chrono::steady_clock::now();
            *c = Transpose(*a) * *b;
            t1 = std::chrono::steady_clock::now();
            timeSpan0 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
            return {false, timeSpan0.count(), 0, c->sum(), 0};
        }

		TestFuncRetType TestForMatTranspose() {
			t0 = std::chrono::steady_clock::now();
			*a = Transpose(*b);
			t1 = std::chrono::steady_clock::now();
			timeSpan0 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(b->data(), M, N,"before transpose: ");
			DEBUG_TOOLS::printRawMatrix(a->data(), M, N, "after transpose: ");
#endif // DEBUG_INFO
#ifdef EIGEN_BENCHMARK
			t2 = std::chrono::steady_clock::now();
			*ea = eb->transpose();
			t3 = std::chrono::steady_clock::now();
			timeSpan1 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
#endif // EIGEN_BENCHMARK
			return { true,timeSpan0.count(),timeSpan1.count(),a->sum(),ea->sum() };
		}

        template <typename Func, typename ...Args>
		LoopFuncRetType Loop(int n, Func&& func, Args... args) {
			double total_t = 0, total_et = 0, sum, sume, maxflops = 0, maxflopse = 0;
            bool flag;
            for (int i = 0; i < n; ++i) {
                double t, et;
                auto counter = (this->*func)(args...);
                std::tie(flag, t, et, sum, sume) = counter;
#ifdef EIGEN_BENCHMARK
                if (!check_by_sum(sum, sume)) {
                    std::cout << "Wrong answer!" << std::endl;
                    std::cout << "Eigen Sum: " << sume << std::endl;
                    std::cout << "Exp Sum: " << sum << std::endl;
					return { 0,0,0,0 };
                }
#endif  // EIGEN_BENCHMARK
				if (flag) {
					total_et += et;
					if (2 * 1e-9 * M * N * K / et > maxflopse)
						maxflopse = 2 * 1e-9 * M * N * K / et;
				}
                total_t += t;
				if (2 * 1e-9 * M * N * K / t > maxflops)
					maxflops = 2 * 1e-9 * M * N * K / t;
				/*system("cls");
				std::cout << "unitTest: " << (i + 1.0f) / n * 100 << '%' << std::endl;*/
            }
			double average_flops = double(2 * 1e-9 * M * N * K / (total_t / n));
			double average_flopse = double(2 * 1e-9 * M * N * K / (total_et / n));

            printTestRes("Matrix Dot Test :" + std::to_string(M) + '*' + std::to_string(K) + '*' + std::to_string(N),
                         double(total_t / n), average_flops);
            if (flag) {
                printTestRes("Eigen Matrix Dot :" + std::to_string(M) + '*' + std::to_string(K) + '*' + std::to_string(N),
                             double(total_et / n), average_flopse);
                check_by_sum(sum, sume, true);
            }
			return { average_flops,average_flopse,maxflops,maxflopse };
        }

        bool check_by_sum(const double& c, const double& ec, bool flag = false) {
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