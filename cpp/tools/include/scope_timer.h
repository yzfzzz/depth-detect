#pragma once
#include <math.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

// 计时器实现全部内联在头文件里：
class ScopedTimer {
  public:
    explicit ScopedTimer(std::string name) :
        name_(std::move(name)),
        start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto   end = std::chrono::steady_clock::now();
        auto   us  = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        auto & timers_table = GetScopedTimers();
        timers_table[name_].push_back(us);
    }

    static std::map<std::string, std::vector<double>> & GetScopedTimers() {
        static std::map<std::string, std::vector<double>> scoped_timers_table;
        return scoped_timers_table;
    }

  private:
    std::string                           name_;
    std::chrono::steady_clock::time_point start_;
};

// 计算百分位数 (P95, P99 等)
inline double calculatePercentile(std::vector<double> data, double percentile) {
    if (data.empty()) {
        return 0.0;
    }
    std::sort(data.begin(), data.end());
    size_t index = static_cast<size_t>(std::ceil(percentile / 100.0 * data.size())) - 1;
    return data[std::min(index, data.size() - 1)] / 1000.0;
}

// 计算平均值
inline double calculateAverage(const std::vector<double> data) {
    if (data.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return (sum / data.size()) / 1000.0;
}

// 基础实现：接受任意可调用对象
template <typename Func>
auto DEBUG_FUNCTION_RUNNING_TIME_IMPL(const std::string & name, Func && func) -> decltype(func()) {
    ScopedTimer timer(name);
    return func();
}

// 版本1：自定义名称 + 可调用对象（适合 lambda）
#define DEBUG_FUNCTION_RUNNING_TIME(name, func) DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, func)

// 版本2：自动函数名 + 可调用对象（适合 lambda）
#define DEBUG_FUNCTION_RUNNING_TIME_AUTO(func) DEBUG_FUNCTION_RUNNING_TIME_IMPL(#func, func)

// 版本3：成员函数专用（对象指针）
#define DEBUG_FUNCTION_RUNNING_TIME_MEMBER_PTR(name, obj, method, ...) \
    DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, [&]() { return (obj)->method(__VA_ARGS__); })

// 版本4：成员函数专用（对象引用）
#define DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, obj, method, ...) \
    DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, [&]() { return (obj).method(__VA_ARGS__); })

// 版本5：自由函数带参数版本
#define DEBUG_FUNCTION_RUNNING_TIME_FUNC(name, func, ...) \
    DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, [&]() { return func(__VA_ARGS__); })
