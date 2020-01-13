#pragma once
#include <chrono>
#include "eigen/Eigen/Dense"
#include <string>
#include <vector>
#include "Matrix.h"
#include "MatrixXpr.h"
#include "Product.h"

#define EIGEN_BENCHMARK

namespace DDA {
    template <typename dtype>
    class Test {
        using myClock = typename std::chrono::steady_clock::time_point;
        using myTimeSpan = typename std::chrono::duration<double>;
        myClock t0;
        myClock t1;
        myTimeSpan timeSpan = std::chrono::duration<double>();

    public:
        void TestForMatDotPerforemence(const int M, const int K, const int N, int numThread) {
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
            timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
            double Gfloaps = 1e-9 * 2 * M * N * K / timeSpan.count();
            printTestRes("Matrix Dot Test :" + std::to_string(M) + '*' + std::to_string(K) + '*' + std::to_string(N),
                         timeSpan.count(), Gfloaps);
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
            t0 = std::chrono::steady_clock::now();
            ec = ea * eb;
            t1 = std::chrono::steady_clock::now();
            timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
            Gfloaps = 1e-9 * 2 * M * N * K / timeSpan.count();
            printTestRes("Eigen Matrix Dot :" + std::to_string(M) + '*' + std::to_string(K) + '*' + std::to_string(N),
                         timeSpan.count(), Gfloaps);
            check_by_sum(c, ec);
#endif
        }

        template <typename myMat, typename EigenMat>
        void check_by_sum(const myMat& c, const EigenMat& ec) {
            std::cout << "Eigen result: " << ec.sum() << std::endl;
            std::cout << "Exp result: " << c.sum() << std::endl;
        }

        void printTestRes(const std::string& s, const double& span, double Gfloaps) {
            std::cout << s << std::endl;
            std::cout << "cost: " << span << std::endl;
            std::cout << "Gfloaps: " << Gfloaps << std::endl;
        }
    };
}  // namespace DDA