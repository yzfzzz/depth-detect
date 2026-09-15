#include "tcp_handler.h"

#include "logger_manager.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
// 发送遇到 EAGAIN 时单次等待可写的超时：对端处理慢不至立刻放弃，
// 又不会拖住主循环超过帧预算的一小部分
constexpr int kSendPollTimeoutMs = 200;
}  // namespace

TcpHandler::TcpHandler(std::string host,
                       int         port,
                       int         check_interval_s,
                       int         connect_timeout_s,
                       bool        auto_reconnect) :
    host_(std::move(host)),
    port_(port),
    check_interval_s_(check_interval_s > 0 ? check_interval_s : 5),
    connect_timeout_s_(connect_timeout_s > 0 ? connect_timeout_s : 3),
    auto_reconnect_(auto_reconnect) {}

TcpHandler::~TcpHandler() {
    stop();
}

void TcpHandler::start() {
    if (auto_reconnect_) {
        // 首连交给看门狗第一轮：状态初始即 BROKEN，请求置位后线程立即尝试
        {
            std::lock_guard<std::mutex> lock(cv_mutex_);
            reconnect_requested_ = true;
        }
        watchdog_ = std::thread(&TcpHandler::watchdogLoop, this);
        APP_INFO("[TcpHandler] watchdog started: probe/reconnect every {} s, connect timeout {} s, "
                 "target {}:{}",
                 check_interval_s_, connect_timeout_s_, host_, port_);
    } else {
        // 启动时同步连接一次，失败则停发（不重连）
        tryReconnect();
        if (state_.load() != ConnState::CONNECTED) {
            APP_WARN("[TcpHandler] initial connect failed for {}:{}, will not send or retry "
                     "(tcp_reconnect: false)",
                     host_, port_);
        }
    }
}

void TcpHandler::stop() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true)) {
        return;  // 已停止（幂等）
    }
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
        cv_.notify_all();  // 立即唤醒看门狗，不等整 5s
    }
    if (watchdog_.joinable()) {
        watchdog_.join();
    }

    // 退出汇总：与落盘/帧统计并排的验收指标
    APP_INFO("[TcpHandler] exit: reconnect ok/fail = {}/{}, alerts dropped while disconnected = {}",
             reconnect_success_.load(), reconnect_fail_.load(), alerts_dropped_.load());

    std::lock_guard<std::mutex> lock(fd_mutex_);
    closeFdLocked();
    state_.store(ConnState::BROKEN);
}

bool TcpHandler::sendJson(const nlohmann::json & j) {
    // 快速路径免锁：非 CONNECTED 直接丢弃（断联期间告警不缓存重发）
    if (state_.load() != ConnState::CONNECTED) {
        alerts_dropped_.fetch_add(1);
        return false;
    }

    const std::string data = j.dump() + "\n";
    size_t            total = 0;
    bool              ok    = true;
    int               err   = 0;
    {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ < 0) {
            ok = false;
        }
        // 保证发完，否则最后的 \n 漏发导致接收区堆积；socket 为非阻塞，
        // EAGAIN 时 poll 等可写后重试，超时则放弃本条（对端处理不过来）
        while (ok && total < data.size()) {
            ssize_t sent = ::send(fd_, data.data() + total, data.size() - total, MSG_NOSIGNAL);
            if (sent > 0) {
                total += static_cast<size_t>(sent);
                continue;
            }
            if (sent < 0 && errno == EINTR) {
                continue;
            }
            if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                struct pollfd pfd;
                pfd.fd      = fd_;
                pfd.events  = POLLOUT;
                pfd.revents = 0;
                if (::poll(&pfd, 1, kSendPollTimeoutMs) > 0) {
                    continue;
                }
            }
            err = errno;
            ok  = false;
        }
    }

    if (!ok) {
        alerts_dropped_.fetch_add(1);
        markBroken(std::string("send failed") + (err != 0 ? (": " + std::string(strerror(err)))
                                                          : std::string()));
        return false;
    }
    return true;
}

void TcpHandler::watchdogLoop() {
    while (!stop_.load()) {
        {
            std::unique_lock<std::mutex> lock(cv_mutex_);
            // 5s 周期探活；发送失败/探活失败置位 reconnect_requested_ 可提前唤醒；
            // stop 立即返回。注意失败重试也走完整周期——服务端宕机时不会忙转
            cv_.wait_for(lock, std::chrono::seconds(check_interval_s_),
                         [this] { return stop_.load() || reconnect_requested_; });
            if (stop_.load()) {
                break;
            }
            reconnect_requested_ = false;
        }
        if (state_.load() == ConnState::CONNECTED) {
            probeConnection();  // 断开则内部转 BROKEN
        }
        if (state_.load() == ConnState::BROKEN) {
            tryReconnect();  // 一次唤醒最多尝试一次（connect 上限 3s）；失败等下一轮
        }
    }
}

void TcpHandler::probeConnection() {
    char      byte    = 0;
    int       err     = 0;
    socklen_t err_len = sizeof(err);
    std::string reason;  // 非空 = 探测判定断连；markBroken 须在锁外调用（其内部要拿 fd_mutex_）
    {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ < 0) {
            reason = "no fd";
        } else {
            // MSG_PEEK 只窥不取（不消费服务端数据，兼容外部 epoll 收数据的约定）；
            // MSG_DONTWAIT 非阻塞，探测成本微秒级、零流量
            ssize_t ret = ::recv(fd_, &byte, sizeof(byte), MSG_PEEK | MSG_DONTWAIT);
            if (ret == 0) {
                reason = "peer closed (FIN)";
            } else if (ret < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                // recv 报错：用 SO_ERROR 区分真断连还是偶发状态
                if (::getsockopt(fd_, SOL_SOCKET, SO_ERROR, &err, &err_len) != 0 || err != 0) {
                    reason = std::string("probe failed: ") + strerror(err != 0 ? err : errno);
                }
                // ret > 0 = 有待读数据但连接活着；ret < 0 + EAGAIN = 无数据、连接正常
            }
        }
    }
    if (!reason.empty()) {
        markBroken(reason);
    }
}

bool TcpHandler::tryReconnect() {
    // CAS 领取任务：同时只有一个线程在 connect，避免 fd 泄漏和双连接
    ConnState expected = ConnState::BROKEN;
    if (!state_.compare_exchange_strong(expected, ConnState::RECONNECTING)) {
        return false;
    }

    // 先清理旧 fd（半开连接留着只占资源），再建新连接
    {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        closeFdLocked();
    }
    int new_fd = doConnectNonBlocking();
    if (new_fd < 0) {
        state_.store(ConnState::BROKEN);
        reconnect_fail_.fetch_add(1);
        APP_WARN("[TcpHandler] reconnect to {}:{} failed (attempt {}), retry in {} s", host_,
                 port_, reconnect_fail_.load(), check_interval_s_);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        fd_ = new_fd;
    }
    state_.store(ConnState::CONNECTED);
    reconnect_success_.fetch_add(1);
    APP_INFO("[TcpHandler] connected to {}:{} (fd = {}, total reconnects ok/fail = {}/{})", host_,
             port_, new_fd, reconnect_success_.load(), reconnect_fail_.load());
    return true;
}

void TcpHandler::markBroken(const std::string & reason) {
    // CAS：只有从 CONNECTED 进入才打日志/计一次断连；BROKEN 下重复调用是幂等的
    ConnState expected = ConnState::CONNECTED;
    if (state_.compare_exchange_strong(expected, ConnState::BROKEN)) {
        APP_WARN("[TcpHandler] connection broken ({}), watchdog will reconnect within {} s",
                 reason, check_interval_s_);
    }
    // 无论状态如何都关掉旧 fd（发送失败时 fd 已不可用）
    {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        closeFdLocked();
    }
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
        reconnect_requested_ = true;  // 免等整 5s，看门狗立即介入
    }
    cv_.notify_one();
}

int TcpHandler::doConnectNonBlocking() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        APP_WARN("[TcpHandler] socket() failed: {}", strerror(errno));
        return -1;
    }
    // 非阻塞 connect：远端不可达时同步 connect 会阻塞上百秒（SYN 重试），
    // 会卡死看门狗；poll 上限保证每轮尝试时间可控
    int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        APP_WARN("[TcpHandler] fcntl(O_NONBLOCK) failed: {}", strerror(errno));
        ::close(fd);
        return -1;
    }

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) <= 0) {
        APP_WARN("[TcpHandler] invalid address: {}", host_);
        ::close(fd);
        return -1;
    }

    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        if (errno != EINPROGRESS) {
            APP_WARN("[TcpHandler] connect() failed: {}", strerror(errno));
            ::close(fd);
            return -1;
        }
        struct pollfd pfd;
        pfd.fd      = fd;
        pfd.events  = POLLOUT;
        pfd.revents = 0;
        int ret     = ::poll(&pfd, 1, connect_timeout_s_ * 1000);
        if (ret <= 0) {
            APP_WARN("[TcpHandler] connect {}:{} timed out after {} s", host_, port_,
                     connect_timeout_s_);
            ::close(fd);
            return -1;
        }
        int       err     = 0;
        socklen_t err_len = sizeof(err);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &err_len) < 0 || err != 0) {
            APP_WARN("[TcpHandler] connect {}:{} failed: {}", host_, port_,
                     strerror(err != 0 ? err : errno));
            ::close(fd);
            return -1;
        }
    }
    return fd;  // 连接建立，保持非阻塞（sendJson 内部处理 EAGAIN）
}

void TcpHandler::closeFdLocked() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}
