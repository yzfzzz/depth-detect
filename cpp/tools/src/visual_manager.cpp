#include "visual_manager.h"

#include "config_manager.h"
#include "logger_manager.h"

// 全局鼠标回调函数
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
    // 调试：检查深度图状态
    if (depth_map_.empty()) {
        std::cout << "[DEBUG] depth_map is empty!" << std::endl;
        return 0.0f;
    }

    std::cout << "[DEBUG] depth_map: size=" << depth_map_.size() << ", type=" << depth_map_.type()
              << ", channels=" << depth_map_.channels() << std::endl;

    const int num_samples = 64;  // 采样点数（5x5网格）
    float     sum_depth   = 0.0f;
    int       valid_count = 0;
    int       zero_count  = 0;

    // BBox 边界
    int left   = static_cast<int>(tlwh[0]);
    int top    = static_cast<int>(tlwh[1]);
    int right  = static_cast<int>(tlwh[0] + tlwh[2]);
    int bottom = static_cast<int>(tlwh[1] + tlwh[3]);

    // 确保边界在图像范围内
    left   = std::max(0, std::min(left, depth_map_.cols - 1));
    top    = std::max(0, std::min(top, depth_map_.rows - 1));
    right  = std::max(0, std::min(right, depth_map_.cols - 1));
    bottom = std::max(0, std::min(bottom, depth_map_.rows - 1));

    // 计算采样步长
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
                // 处理其他类型
                std::cout << "[DEBUG] Unsupported depth map type: " << depth_map_.type()
                          << std::endl;
            }

            if (depth == 0) {
                zero_count++;
            } else if (depth > 0) {
                sum_depth += depth;
                valid_count++;
            }
        }
    }

    std::cout << "[DEBUG] Samples: total=" << num_samples << ", valid=" << valid_count
              << ", zero=" << zero_count << std::endl;

    return (valid_count > 0) ? (sum_depth / valid_count) : 0.0f;
}

void DisplayManager::printTargetInfo(const STrack & track) const {
    int                        class_id = track.class_id_;
    int                        track_id = track.track_id_;
    const std::vector<float> & tlwh     = track.tlwh_;

    // 使用多点采样计算深度均值
    float depth = computeMeanDepth(tlwh);

    std::cout << "\n=== Target Info ===" << std::endl;
    std::cout << "Class: " << V_CLASS_NAMES[class_id] << std::endl;
    std::cout << "Track ID: " << track_id << std::endl;
    std::cout << "Depth (mean of 25 samples): " << depth << std::endl;
    std::cout << "BBox: [" << tlwh[0] << ", " << tlwh[1] << ", " << tlwh[0] + tlwh[2] << ", "
              << tlwh[1] + tlwh[3] << "]" << std::endl;
    std::cout << "==================\n" << std::endl;
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
    std::cout << "No target clicked" << std::endl;
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
        std::cout << "Paused, press SPACE to continue..." << std::endl;
        while (true) {
            char pause_key = cv::waitKey(0);
            if (pause_key == Key_Input::SPACE) {
                std::cout << "Resuming..." << std::endl;
                break;
            } else if (pause_key == Key_Input::ESC) {  // ESC键退出
                std::cout << "User exited" << std::endl;
                return Key_Input::ESC;
            }
        }
        return 0;
    } else if (key == Key_Input::ESC) {  // ESC键退出
        std::cout << "User exited" << std::endl;
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

void DrawingManager::drawTrackedObject(cv::Mat &                     img,
                                       const STrack &                track,
                                       const MotionStateInfoRecord & motion_state,
                                       cv::Scalar                    color) {
    const std::vector<float> & tlwh     = track.tlwh_;
    int                        class_id = track.class_id_;
    int                        track_id = track.track_id_;

    // 1. 获取运动状态字符串
    auto        it = MOTION_STR_MAP.find({ motion_state.state_vec, motion_state.state_acc });
    std::string motion_state_str = (it != MOTION_STR_MAP.end()) ? it->second : "Unknown";

    // 2. 准备文字标签
    std::string label = cv::format("%s #%d [%s]", vClassNames_[class_id].c_str(), track_id,
                                   motion_state_str.c_str());

    // 3. 绘制文字背景框和文字
    int      baseLine   = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
    cv::Rect rect_bg(cv::Point((int) tlwh[0], (int) tlwh[1] - label_size.height - 8),
                     cv::Size(label_size.width + 8, label_size.height + 8));

    cv::rectangle(img, rect_bg, color, cv::FILLED);
    cv::putText(img, label, cv::Point((int) tlwh[0] + 4, (int) tlwh[1] - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // 4. 绘制目标主体矩形框
    cv::rectangle(img, cv::Rect((int) tlwh[0], (int) tlwh[1], (int) tlwh[2], (int) tlwh[3]), color,
                  2);

    // 5. 检查运动状态是否为"加速靠近"，如果是则绘制红色交叉
    if (motion_state.state_vec == MotionState::APPROACH &&
        motion_state.state_acc == MotionState::ACCELE) {
        int x1 = static_cast<int>(tlwh[0]);
        int y1 = static_cast<int>(tlwh[1]);
        int x2 = static_cast<int>(tlwh[0] + tlwh[2]);
        int y2 = static_cast<int>(tlwh[1] + tlwh[3]);

        cv::Scalar red_color(0, 0, 255);  // BGR 格式的红色
        int        line_thickness = 2;

        // 绘制左上到右下的线
        cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), red_color, line_thickness, cv::LINE_AA);
        // 绘制左下到右上的线
        cv::line(img, cv::Point(x1, y2), cv::Point(x2, y1), red_color, line_thickness, cv::LINE_AA);
    }
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
