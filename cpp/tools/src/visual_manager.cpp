#include "visual_manager.h"

#include "config_manager.h"
#include "logger_manager.h"

// OpenCV 全局鼠标回调：将点击事件转发给对应窗口的 DisplayManager 实例
// userdata 必须是 DisplayManager 指针，由 cv::setMouseCallback 设置
void onMouse(int event, int x, int y, int flags, void * userdata) {
    DisplayManager * dm = static_cast<DisplayManager *>(userdata);
    if (!dm || !dm->isEnabled()) {
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        dm->handleMouseClick(x, y);
    }
}

DisplayManager::DisplayManager(const ConfigManager & config,
                               const std::string &   window_name,
                               cv::Size              display_size) :
    enabled_(config.isDisplayEnabled()),
    window_name_(window_name),
    display_size_(display_size) {
    if (enabled_) {
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, display_size_.width, display_size_.height);
        cv::setMouseCallback(window_name_, onMouse, this);
        APP_INFO("DisplayManager initialized");
    }
}

DisplayManager::~DisplayManager() {
    if (enabled_) {
        cv::destroyWindow(window_name_);
    }
}

float DisplayManager::computeMeanDepth(const std::vector<float> & tlwh) const {
    // 交互式深度查询的简化版均值计算：均匀网格采样 + 简单平均
    // 注意：此函数仅用于鼠标点击时显示目标信息，精度要求低于运动状态引擎中的截断均值法
    if (depth_map_.empty()) {
        APP_INFO("depth_map is empty!");
        return 0.0f;
    }

    const int num_samples = 64;  // 采样点数（5x5网格）
    float     sum_depth   = 0.0f;
    int       valid_count = 0;
    int       zero_count  = 0;

    // BBox 边界钳位到深度图范围内，防止越界访问
    int left   = static_cast<int>(tlwh[0]);
    int top    = static_cast<int>(tlwh[1]);
    int right  = static_cast<int>(tlwh[0] + tlwh[2]);
    int bottom = static_cast<int>(tlwh[1] + tlwh[3]);

    // 确保边界在图像范围内
    left   = std::max(0, std::min(left, depth_map_.cols - 1));
    top    = std::max(0, std::min(top, depth_map_.rows - 1));
    right  = std::max(0, std::min(right, depth_map_.cols - 1));
    bottom = std::max(0, std::min(bottom, depth_map_.rows - 1));

    int width  = right - left;
    int height = bottom - top;

    if (width <= 0 || height <= 0) {
        return 0.0f;
    }

    // 在 BBox 内均匀采样（5x5 网格）
    int   grid_size = static_cast<int>(std::sqrt(num_samples));
    float step_x    = static_cast<float>(width) / (grid_size - 1);
    float step_y    = static_cast<float>(height) / (grid_size - 1);

    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int x = left + static_cast<int>(i * step_x);
            int y = top + static_cast<int>(j * step_y);

            // 边界检查
            if (x < 0 || x >= depth_map_.cols || y < 0 || y >= depth_map_.rows) {
                continue;
            }

            float depth = 0.0f;
            if (depth_map_.type() == CV_32FC1) {
                depth = depth_map_.at<float>(y, x);
            } else if (depth_map_.type() == CV_8UC1) {
                depth = static_cast<float>(depth_map_.at<uchar>(y, x));
            } else {
                APP_INFO("Unsupported depth map type: {}", depth_map_.type());
                continue;  // 跳过不支持的深度图格式
            }

            if (depth > 0) {
                sum_depth += depth;
                valid_count++;
            }
        }
    }

    APP_INFO("Depth samples: total={}, valid={}", num_samples, valid_count);

    return (valid_count > 0) ? (sum_depth / valid_count) : 0.0f;
}

void DisplayManager::printTargetInfo(const STrack & track) const {
    int                        class_id = track.class_id_;
    int                        track_id = track.track_id_;
    const std::vector<float> & tlwh     = track.tlwh_;

    // 使用多点采样计算深度均值
    float depth = computeMeanDepth(tlwh);

    APP_INFO("\n=== Target Info ===");
    APP_INFO("Class: {}", V_CLASS_NAMES[class_id]);
    APP_INFO("Track ID: {}", track_id);
    APP_INFO("Depth (mean of 25 samples): {}", depth);
    APP_INFO("BBox: [{}, {}, {}, {}]", tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]);
    APP_INFO("==================");
}

void DisplayManager::handleMouseClick(int x, int y) {
    for (const auto & track : tracks_) {
        const std::vector<float> & tlwh   = track.tlwh_;
        float                      left   = tlwh[0];
        float                      top    = tlwh[1];
        float                      right  = tlwh[0] + tlwh[2];
        float                      bottom = tlwh[1] + tlwh[3];

        if (x >= left && x <= right && y >= top && y <= bottom) {
            printTargetInfo(track);
            return;
        }
    }
    APP_WARN("No target clicked");
}

void DisplayManager::updateData(const std::vector<STrack> & tracks, const cv::Mat & depth_map) {
    if (enabled_) {
        tracks_ = tracks;
        if (!depth_map.empty()) {
            depth_map_ = depth_map.clone();
            cv::resize(depth_map_, depth_map_, cv::Size(1280, 720));
        }
    }
}

int DisplayManager::handleKey(int key) {
    if (!enabled_) {
        return key;  // 返回原始按键
    }

    if (key == Key_Input::SPACE) {
        APP_INFO("Paused, press SPACE to continue...");
        while (true) {
            char pause_key = cv::waitKey(0);
            if (pause_key == Key_Input::SPACE) {
                APP_INFO("Resuming...");
                break;
            } else if (pause_key == Key_Input::ESC) {  // ESC键退出
                APP_INFO("User exited");
                return Key_Input::ESC;
            }
        }
        return 0;
    } else if (key == Key_Input::ESC) {  // ESC键退出
        APP_INFO("User exited");
        return Key_Input::ESC;
    }
    return key;
}

void DisplayManager::show(const cv::Mat & frame) {
    if (enabled_) {
        cv::imshow(window_name_, frame);
    }
}

int DisplayManager::waitKey(int delay) {
    if (!enabled_) {
        return Key_Input::ESC;
    }
    return cv::waitKey(delay);
}

DrawingManager::DrawingManager(const std::vector<std::string> & class_names) :
    vClassNames_(class_names) {}

void DrawingManager::drawTrackedObject(cv::Mat &            img,
                                       const STrack &       track,
                                       const AlertMessage & alert_msg,
                                       cv::Scalar           color) {
    const std::vector<float> & tlwh     = track.tlwh_;
    int                        class_id = track.class_id_;
    int                        track_id = track.track_id_;

    // 准备文字标签
    std::string label = cv::format("%s #%d", vClassNames_[class_id].c_str(), track_id);

    // 绘制文字背景框和文字
    int      baseLine   = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
    cv::Rect rect_bg(cv::Point((int) tlwh[0], (int) tlwh[1] - label_size.height - 8),
                     cv::Size(label_size.width + 8, label_size.height + 8));

    // 绘制目标主体矩形框
    // 检查物体是否危险
    bool is_danger = false;
    for (size_t i = 0; i < alert_msg.objects.size(); ++i) {
        if (alert_msg.objects[i].track_id == track_id) {
            is_danger = alert_msg.objects[i].is_danger;
            break;
        }
    }

    int x1 = static_cast<int>(tlwh[0]);
    int y1 = static_cast<int>(tlwh[1]);
    int x2 = static_cast<int>(tlwh[0] + tlwh[2]);
    int y2 = static_cast<int>(tlwh[1] + tlwh[3]);

    // 裁剪到图像范围内，防止 ROI 越界
    x1 = std::max(0, std::min(x1, img.cols - 1));
    y1 = std::max(0, std::min(y1, img.rows - 1));
    x2 = std::max(0, std::min(x2, img.cols - 1));
    y2 = std::max(0, std::min(y2, img.rows - 1));

    int w = x2 - x1;
    int h = y2 - y1;

    if (is_danger && w > 0 && h > 0) {
        // 半透明红色填充 (alpha ≈ 0.3)
        cv::Mat roi = img(cv::Rect(x1, y1, w, h));
        cv::Mat red_overlay(roi.size(), roi.type(), cv::Scalar(0, 0, 255));
        cv::addWeighted(red_overlay, 0.2, roi, 0.7, 0, roi);

        // 红色边框（用原始未裁剪的 bbox 绘制，保持视觉一致）
        color = cv::Scalar(0, 0, 255);
    }
    cv::rectangle(img, cv::Rect(x1, y1, w, h), color, 2);
    cv::rectangle(img, rect_bg, color, cv::FILLED);
    cv::putText(img, label, cv::Point((int) tlwh[0] + 4, (int) tlwh[1] - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

void DrawingManager::drawGlobalInfo(cv::Mat & img,
                                    int       num_frames,
                                    int       show_fps,
                                    size_t    num_tracks) {
    cv::putText(img, cv::format("frame: %d fps: %d num: %zu", num_frames, show_fps, num_tracks),
                cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2,
                cv::LINE_AA);
}

cv::Mat DrawingManager::concatenateFrames(const cv::Mat & rgb_img, const cv::Mat & depth_vis) {
    cv::Mat out_frame;

    // 如果没有深度图，直接返回原图的一份拷贝
    if (depth_vis.empty()) {
        rgb_img.copyTo(out_frame);
        return out_frame;
    }

    // 确保深度可视化的宽度高度跟原图一致，避免拼接越界崩溃
    cv::Mat resized_depth_vis;
    if (depth_vis.size() != rgb_img.size()) {
        cv::resize(depth_vis, resized_depth_vis, rgb_img.size());
    } else {
        resized_depth_vis = depth_vis;
    }

    // 创建 (原图高度 + 深度图高度) 作为总高度，宽度不变
    out_frame.create(rgb_img.rows + resized_depth_vis.rows, rgb_img.cols, rgb_img.type());

    // 拷贝原图到上半部分
    rgb_img.copyTo(out_frame(cv::Rect(0, 0, rgb_img.cols, rgb_img.rows)));

    // 拷贝深度图到下半部分
    resized_depth_vis.copyTo(
        out_frame(cv::Rect(0, rgb_img.rows, rgb_img.cols, resized_depth_vis.rows)));

    return out_frame;
}
