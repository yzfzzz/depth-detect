#pragma once

// 通用小工具集合：header-only、不引入第三方依赖

// 两个非负整数的最大公约数
inline int gcdInt(int a, int b) {
    while (b != 0) {
        const int t = a % b;
        a           = b;
        b           = t;
    }
    return a;
}

// 两个正整数的最小公倍数
// 错峰调度里用到：轻量检测模型只在检测与深度撞车的帧被调用，其调用周期 = 两个间隔的最小公倍数
inline int lcmInt(int a, int b) {
    return a / gcdInt(a, b) * b;
}
