#include "BYTETracker.h"
#include "cvnp/cvnp.h"
#include "depth_model.h"
#include "frame.h"
#include "motion_state_engine.h"
#include "STrack.h"
#include "yolo_detect_model.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;

// ━━━━━ 工具函数：numpy 图片 → FrameInputContext（含 GPU 上传）━━━━━
FrameInputContext make_frame_context(py::array_t<uint8_t> & img, int frame_id, double fps) {
    cv::Mat           mat = cvnp::nparray_to_mat(img);
    FrameMeta         meta(mat.cols, mat.rows, fps, FrameSource::VIDEO);
    FrameInputContext ctx(frame_id, meta);
    ctx.raw_img = mat;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        void * ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, ctx.img_size));
        ctx.d_raw_img_.reset(static_cast<uchar *>(ptr));
        CHECK_CUDA(
            cudaMemcpy(ctx.d_raw_img_.get(), mat.data, ctx.img_size, cudaMemcpyHostToDevice));
    }
    return ctx;
}

// ==================== 绑定 Detection 结构体 ====================
void bind_detection(py::module & m) {
    py::class_<Detection>(m, "Detection")
        .def_readonly("bbox", &Detection::bbox)
        .def_readwrite("conf", &Detection::conf)
        .def_readwrite("class_id", &Detection::classId)
        .def("__repr__", [](const Detection & d) {
            return py::str(
                       "Detection(bbox=[{:.1f}, {:.1f}, {:.1f}, {:.1f}], conf={:.2f}, class_id={})")
                .format(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.conf, d.classId);
        });
}

// ==================== 绑定 MotionState 枚举 ====================
void bind_motion_state(py::module & m) {
    py::enum_<MotionState>(m, "MotionState")
        .value("INVALID", MotionState::INVAILD)
        .value("UNKNOWN", MotionState::UNKNOWN)
        .value("STABLE", MotionState::STABLE)
        .value("APPROACH", MotionState::APPROACH)
        .value("MOVE_AWAY", MotionState::MOVE_AWAY)
        .value("ACCELE", MotionState::ACCELE)
        .value("DECELE", MotionState::DECELE)
        .value("CONSTANT", MotionState::CONSTANT)
        .export_values();
}

// ==================== 绑定 MotionStateInfoRecord 结构体 ====================
void bind_motion_state_info_record(py::module & m) {
    py::class_<MotionStateInfoRecord>(m, "MotionStateInfoRecord")
        .def_readonly("state_vec", &MotionStateInfoRecord::state_vec)
        .def_readonly("state_acc", &MotionStateInfoRecord::state_acc)
        .def_readonly("velocity", &MotionStateInfoRecord::velocity)
        .def(py::init<MotionState, MotionState, float>())
        .def("__repr__", [](const MotionStateInfoRecord & r) {
            return py::str("MotionStateInfoRecord(state_vec={}, state_acc={}, velocity={:.2f})")
                .format((int) r.state_vec, (int) r.state_acc, r.velocity);
        });
}

// ==================== 绑定 YoloDetectModel ====================
void bind_yolo_detector(py::module & m) {
    py::class_<YoloDetectModel>(m, "YoloDetectModel")
        .def(py::init<>())
        .def("init", &YoloDetectModel::init, py::arg("model_path"), py::arg("raw_img_w"),
             py::arg("raw_img_h"), py::arg("nms_thresh"), py::arg("conf_thresh"),
             py::arg("num_class"), py::arg("use_gpu") = false,
             "Initialize YOLO model with paths and parameters")
        // 便捷方法：直接接受 numpy 图片，内部管理 FrameInputContext
        .def(
            "inference",
            [](YoloDetectModel & self, py::array_t<uint8_t> & img, int frame_id, double fps) {
                FrameInputContext  ctx = make_frame_context(img, frame_id, fps);
                InferOutputContext out;

                self.runInferenceAsync(ctx);
                self.getInferOutputResult(out);
                return out.detections;
            },
            py::arg("img"), py::arg("frame_id") = 0, py::arg("fps") = 30.0,
            "Run YOLO inference on a numpy image (HxWxC, uint8). "
            "Returns list of Detection objects.");
}

// ==================== 绑定 DepthModel（统一的深度模型）====================
void bind_depth_models(py::module & m) {
    py::class_<DepthModel>(m, "DepthModel")
        .def(py::init<>())
        .def("init", &DepthModel::init, py::arg("model_path"), py::arg("raw_img_w"),
             py::arg("raw_img_h"), py::arg("is_normalize") = false, py::arg("use_gpu") = false,
             "Initialize the depth model with given paths. "
             "model_path is a dict with 'engine' and/or 'onnx' keys.")
        // 便捷方法：直接接受 numpy 图片，返回 (depth_map, depth_vis) 两个 numpy 数组
        .def(
            "predict",
            [](DepthModel & self, py::array_t<uint8_t> & img, int frame_id, double fps) {
                FrameInputContext  ctx = make_frame_context(img, frame_id, fps);
                InferOutputContext out;
                self.runInferenceAsync(ctx);
                self.getInferOutputResult(out);
                // clone 确保数据独立于模型内部的 pinned memory
                return py::make_tuple(out.result_depth.clone(), out.depth_vis.clone());
            },
            py::arg("img"), py::arg("frame_id") = 0, py::arg("fps") = 30.0,
            "Run depth prediction on a numpy image. "
            "Returns (depth_map, depth_colormap) as numpy arrays.");
}

// ==================== 绑定 MotionStateEngine ====================
void bind_motion_state_engine(py::module & m) {
    py::class_<MotionStateEngine>(m, "MotionStateEngine")
        .def(py::init<float, float, float, float>(), py::arg("velocity_threshold") = 5.0f,
             py::arg("acceleration_threshold") = 1.5f, py::arg("kf_process_noise_cov") = 2e-2f,
             py::arg("kf_measurement_noise_cov") = 5e-2f)
        .def("compute_motion_state", &MotionStateEngine::computeMotionState, py::arg("track_id"),
             py::arg("raw_depth"), py::arg("timestamp"), "Compute motion state using Kalman filter")
        .def("get_object_depth", &MotionStateEngine::getObjectDepth, py::arg("depth"),
             py::arg("track"), py::arg("image_size"));
}

// ==================== 绑定 BYTETracker ====================
void bind_byte_tracker(py::module & m) {
    py::class_<Object>(m, "Object")
        .def(py::init<>())
        .def_property(
            "rect",
            [](const Object & self) {
                return py::make_tuple(self.rect.x, self.rect.y, self.rect.width, self.rect.height);
            },
            [](Object & self, py::sequence seq) {
                if (py::len(seq) != 4) {
                    throw std::runtime_error(
                        "Object.rect must be a sequence of 4 elements: [x, y, w, h]");
                }
                self.rect.x      = py::cast<float>(seq[0]);
                self.rect.y      = py::cast<float>(seq[1]);
                self.rect.width  = py::cast<float>(seq[2]);
                self.rect.height = py::cast<float>(seq[3]);
            })
        .def_readwrite("label", &Object::label)
        .def_readwrite("prob", &Object::prob)
        .def_readwrite("distance", &Object::distance);

    py::class_<STrack>(m, "STrack")
        .def_readonly("tlwh", &STrack::tlwh_)
        .def_readonly("track_id", &STrack::track_id_)
        .def_readonly("class_id", &STrack::class_id_)
        .def_readonly("score", &STrack::score_)
        .def_readonly("is_activated", &STrack::is_activated_);

    py::class_<BYTETracker>(m, "BYTETracker")
        .def(py::init<int, int>(), py::arg("frame_rate") = 30, py::arg("track_buffer") = 30)
        .def("update", &BYTETracker::update, py::arg("objects"),
             "Update tracker with new detections");
}

// ==================== 导出常量 ====================
void bind_constants(py::module & m) {
    m.attr("MOTION_STR_MAP") = MOTION_STR_MAP;
}

// ==================== 主模块定义 ====================
PYBIND11_MODULE(depth_detection, m) {
    m.doc() = "Depth Detection Python Bindings — inference-only C++ bridge to Python";

    // ── 初始化 spdlog 默认 logger，防止 APP_INFO/APP_ERROR 空指针崩溃 ──
    auto logger = spdlog::get("app");
    if (!logger) {
        logger = spdlog::stdout_color_mt("app");
        spdlog::set_level(spdlog::level::info);  // Python 侧只显示 warn 以上，减少刷屏
    }

    bind_detection(m);
    bind_depth_models(m);
    bind_motion_state(m);
    bind_motion_state_info_record(m);
    bind_yolo_detector(m);
    bind_motion_state_engine(m);
    bind_byte_tracker(m);
    bind_constants(m);
}
