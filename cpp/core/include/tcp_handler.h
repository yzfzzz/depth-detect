#pragma once

#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

// TCP 断联自动重连管理器（Linux，POSIX socket）
//
// 架构：主循环线程只做 sendJson（状态感知、非阻塞、微秒级临界区）；
// 看门狗线程每 check_interval_s 醒来探活，断开后自动重连（单次 connect
// 上限 connect_timeout_s，绝不阻塞主循环）。
//
// 断连检测三路信号：
//   1. 看门狗周期探活：recv(MSG_PEEK|MSG_DONTWAIT) == 0 → 对端已 FIN
//   2. 看门狗周期探活：getsockopt(SO_ERROR) != 0 → 连接已被 RST
//   3. 发送失败（EPIPE/ECONNRESET/EAGAIN 超时）→ 立即置 BROKEN 并唤醒看门狗
//
// 状态机：CONNECTED --探活/发送失败--> BROKEN --看门狗领取--> RECONNECTING
//         --connect 成功--> CONNECTED / --失败--> BROKEN（等下一轮）。
// fd 的 send/close 互斥在一把锁内；新连接"先建好后替换"，发送方要么用旧 fd
// 发完、要么看到 BROKEN 直接丢弃，永远不会碰到半成品连接。
class TcpHandler {
  public:
    enum class ConnState { CONNECTED, BROKEN, RECONNECTING };

    // auto_reconnect=false 时保持旧行为：仅启动时同步连接一次，失败后不再重试
    TcpHandler(std::string host,
               int         port,
               int         check_interval_s = 5,
               int         connect_timeout_s = 3,
               bool        auto_reconnect   = true);
    ~TcpHandler();  // stop + join + close，不得抛出

    TcpHandler(const TcpHandler &)            = delete;
    TcpHandler & operator=(const TcpHandler &) = delete;

    // 拉起看门狗线程（auto_reconnect）或做一次性首连（否则）。Init 时调用
    void start();
    // 置停止标志、唤醒看门狗、join、关 fd。析构前调用；可重入
    void stop();

    // 状态感知发送：CONNECTED 才进发送临界区；BROKEN 直接丢弃并计数。
    // 返回 false 表示本条未发出（断联期间告警不缓存重发——过时报警重放
    // 可能误导下游，宁缺勿错）
    bool sendJson(const nlohmann::json & j);

    bool     isConnected() const { return state_.load() == ConnState::CONNECTED; }
    uint64_t reconnectSuccessCount() const { return reconnect_success_.load(); }
/**
 * 获取重连失败次数的计数器值
 * @return 返回重连失败的次数，类型为uint64_t
 */
    uint64_t reconnectFailCount() const { return reconnect_fail_.load(); }
    uint64_t alertsDroppedCount() const { return alerts_dropped_.load(); }

  private:
    // 看门狗主循环：5s 周期探活；断连后每轮尝试重连一次（connect 上限 3s）
    void watchdogLoop();
    // 探活：recv(PEEK|DONTWAIT) 判 FIN + SO_ERROR 判 RST；断开则 markBroken
    void probeConnection();
    // 领取重连任务：CAS BROKEN->RECONNECTING（串行化），非阻塞 connect + poll
    bool tryReconnect();
    // 标记断连：CAS CONNECTED->BROKEN、关旧 fd、请求看门狗立即介入
    void markBroken(const std::string & reason);
    // 非阻塞 connect：socket(O_NONBLOCK) -> connect(EINPROGRESS) ->
    // poll(POLLOUT, connect_timeout_s) -> SO_ERROR 验证。成功返回 fd，失败 -1
    int  doConnectNonBlocking();
    // 关闭当前 fd（幂等，须持 fd_mutex_ 或在停止后调用）
    void closeFdLocked();

    std::string host_;
    int         port_;
    int         check_interval_s_;   // 看门狗探活/重试周期
    int         connect_timeout_s_;  // 单次非阻塞 connect 的 poll 上限
    bool        auto_reconnect_;

    std::atomic<ConnState> state_{ ConnState::BROKEN };

    std::mutex fd_mutex_;  // 保护 fd_ 的所有 syscall（send/connect 替换/close）
    int        fd_ = -1;   // 当前连接；-1 = 无连接

    std::atomic<bool>   stop_{ false };
    std::thread         watchdog_;
    std::mutex          cv_mutex_;  // 保护 reconnect_requested_，配对 cv_
    std::condition_variable cv_;
    bool reconnect_requested_ = false;  // 发送失败/探活失败时置位，免等整 5s

    // 运行统计（跨线程，atomic）
    std::atomic<uint64_t> reconnect_success_{ 0 };
    std::atomic<uint64_t> reconnect_fail_{ 0 };
    std::atomic<uint64_t> alerts_dropped_{ 0 };
};
