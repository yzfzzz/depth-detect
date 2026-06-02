#include "BYTETracker.h"
#include "cvnp/cvnp.h"
#include "depth_anything.h"
#include "lite_mono.h"
#include "motion_state_engine.h"
#include "STrack.h"
#include "types.h"
#include "yolo_detect_model.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ==================== 绑定 Detection 结构体 ====================
void bind_detection(py::module & m) {
    py::class_<Detection>(m, "Detection")
        .def_readonly("bbox", &Detection::bbox)
        .def_readwrite("conf", &Detection::conf)
        .def_readwrite("class_id", &Detection::classId)
        .def("__repr__", [](const Detection & d) {
            return py::str("Detection(bbox={}, conf={:.2f}, class_id={})")
                .format(d.bbox, d.conf, d.classId);
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
        .def(py::init<const std::string &, int, float, float, int>(), py::arg("trt_file"),
             py::arg("gpu_id") = 0, py::arg("nms_thresh") = 0.45f, py::arg("conf_thresh") = 0.25f,
             py::arg("num_class") = 80)
        .def("inference", &YoloDetectModel::inference, py::arg("img"),
             "Run inference on the input image and return a list of Detection results");
}

// ==================== 绑定 depth深度检测模型 ====================
void bind_depth_models(py::module & m) {
    py::class_<DepthModel, std::shared_ptr<DepthModel>>(m, "DepthModel")
        .def("init", &DepthModel::init, py::arg("engine_path"),
             "Initialize the depth model with the given engine path")
        .def("predict", &DepthModel::predict, py::arg("image"), "Run depth prediction");

    py::class_<DepthAnything, DepthModel, std::shared_ptr<DepthAnything>>(m, "DepthAnything")
        .def(py::init<>());

    py::class_<LiteMono, DepthModel, std::shared_ptr<LiteMono>>(m, "LiteMono").def(py::init<>());
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
    // Object 结构体

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

    // STrack
    py::class_<STrack>(m, "STrack")
        .def_readonly("tlwh", &STrack::tlwh_)
        .def_readonly("track_id", &STrack::track_id_)
        .def_readonly("class_id", &STrack::class_id_)
        .def_readonly("score", &STrack::score_)
        .def_readonly("is_activated", &STrack::is_activated_);

    // BYTETracker
    py::class_<BYTETracker>(m, "BYTETracker")
        .def(py::init<int, int>(), py::arg("frame_rate") = 30, py::arg("track_buffer") = 30)
        .def(
            "update",
            [](BYTETracker & self, std::vector<Object> & objects) { return self.update(objects); },
            py::arg("objects"), "Update tracker with new detections");
}

// ==================== 导出常量 ====================
void bind_constants(py::module & m) {
    m.attr("MOTION_STR_MAP") = MOTION_STR_MAP;
}

// ==================== 主模块定义 ====================
PYBIND11_MODULE(depth_detection, m) {
    m.doc() = "Depth Detection Python Bindings - C++ core classes exposed to Python";

    // 绑定各类
    bind_detection(m);
    bind_depth_models(m);
    bind_motion_state(m);
    bind_motion_state_info_record(m);
    bind_yolo_detector(m);
    bind_motion_state_engine(m);
    bind_byte_tracker(m);
    bind_constants(m);
}
