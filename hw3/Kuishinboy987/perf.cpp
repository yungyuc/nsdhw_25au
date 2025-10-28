#include "mul_set.hpp"
#include <chrono>
#include <fstream>
#include <cstdlib>

static void fill(Matrix& M) {
    for (size_t i=0;i<M.nrow();++i)
        for (size_t j=0;j<M.ncol();++j)
            M(i,j) = 0.001 * double(i) + 0.002 * double(j);
}

static volatile uint8_t sink_byte = 0;

// 以走訪大 buffer 的方式「汙染」快取（近似清快取）
static void cache_flush(size_t bytes = (size_t)256 << 20) { // 預設 256 MiB
    static std::vector<uint8_t> buf;
    if (buf.size() < bytes) buf.resize(bytes, 1);
    for (size_t i = 0; i < bytes; i += 64) {  // 以 cache line 為步長
        sink_byte ^= buf[i];
    }
}

int main() {
    const size_t N = []{
        const char* s = std::getenv("SIZE");
        return s ? static_cast<size_t>(std::stoul(s)) : size_t(1000);
    }();
    const size_t T = []{
        const char* s = std::getenv("TILE");
        return s ? static_cast<size_t>(std::stoul(s)) : size_t(100);
    }();

    Matrix A(N,N), B(N,N);
    fill(A); fill(B);

    cache_flush();
    auto t0 = std::chrono::high_resolution_clock::now();
    auto C1 = multiply_naive(A,B);
    auto t1 = std::chrono::high_resolution_clock::now();

    cache_flush();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto C2 = multiply_tile(A,B,T);
    auto t3 = std::chrono::high_resolution_clock::now();

    cache_flush();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto C3 = multiply_mkl(A,B);
    auto t5 = std::chrono::high_resolution_clock::now();

    auto ms_naive = std::chrono::duration<double, std::milli>(t1-t0).count();
    auto ms_tile  = std::chrono::duration<double, std::milli>(t3-t2).count();
    auto ms_mkl   = std::chrono::duration<double, std::milli>(t5-t4).count();

    std::ofstream ofs("performance.txt");
    ofs << "N=" << N << ", TILE=" << T << "\n";
    ofs << "naive(ms): " << ms_naive << "\n";
    ofs << "tile(ms):  " << ms_tile  << "  (tile/naive = " << (ms_tile/ms_naive) << ")\n";
    ofs << "mkl(ms):   " << ms_mkl   << "  (mkl/naive  = " << (ms_mkl/ms_naive)  << ")\n";
    ofs.close();

    return 0;
}
