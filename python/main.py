import cv2
import yaml
from pathlib import Path
import argparse
import depth_detection as dd


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_engine_path(model_path_list: list) -> str:
    """从 config 的 model_path 列表中提取 engine 文件路径"""
    for item in model_path_list:
        if item.get("type") == "engine":
            return item["path"]
    # 回退到 onnx
    for item in model_path_list:
        if item.get("type") == "onnx":
            return item["path"]
    raise ValueError("No engine or onnx path found in model_path")


def build_depth_model(model_path_list: list, frame_w: int, frame_h: int, use_gpu: bool) -> dd.DepthModel:
    """统一创建 DepthModel，不再区分 LiteMono / DepthAnything 子类"""
    model = dd.DepthModel()
    model_path = {item["type"]: item["path"] for item in model_path_list}
    # LiteMono 不需要 normalize，DepthAnything 需要 — 从路径名推断
    engine_name = Path(extract_engine_path(model_path_list)).name.lower()
    is_normalize = "lite" not in engine_name
    model.init(model_path, frame_w, frame_h, is_normalize, use_gpu)
    return model


def draw_track_overlay(frame, track, motion_state, raw_depth, state_str):
    tlwh = list(track.tlwh)
    x, y, w, h = map(int, tlwh)
    x2, y2 = x + w, y + h

    is_accelerate_approach = (
        motion_state.state_vec == dd.MotionState.APPROACH
        and motion_state.state_acc == dd.MotionState.ACCELE
    )

    base_color = (0, 0, 255) if is_accelerate_approach else (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 半透明底色
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), base_color, thickness=-1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # 边框
    cv2.rectangle(frame, (x, y), (x2, y2), base_color, 2, cv2.LINE_AA)

    # 标签文字
    label_text = f"ID {track.track_id} D={raw_depth:.2f}"
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    label_x1 = x
    label_y1 = max(0, y - text_h - baseline - 8)
    label_x2 = x + text_w + 10
    label_y2 = y

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), base_color, -1)
    cv2.putText(
        frame,
        label_text,
        (x + 5, y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # 框下状态
    cv2.putText(
        frame,
        state_str,
        (x, y2 + 18),
        font,
        0.5,
        base_color,
        1,
        cv2.LINE_AA,
    )

    # 加速靠近时画红色交叉
    if is_accelerate_approach:
        pad = max(4, min(w, h) // 8)
        cross_color = (0, 0, 255)
        cv2.line(frame, (x + pad, y + pad), (x2 - pad, y2 - pad), cross_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + pad, y2 - pad), (x2 - pad, y + pad), cross_color, 2, cv2.LINE_AA)

    return frame


def parse_args():
    root_dir = Path(__file__).resolve().parent
    default_config = root_dir.parent / "bin" / "config.yaml"
    default_video = root_dir.parent / "data" /  "1shu_east_0514.mp4"
    print(f"Default config: {default_config}")
    print(f"Default video: {default_video}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config, help="path to config.yaml")
    parser.add_argument("--video", type=Path, default=default_video, help="path to input video")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    yolo_cfg = config.get("yolo", {})
    depth_cfg = config.get("depth", {})
    motion_cfg = config.get("motion_state_engine", {})
    display_cfg = config.get("display_manager", {})
    prefer_cfg = config.get("prefer", {})

    yolo_model_path = yolo_cfg["yolo_model_path"]
    depth_model_path = depth_cfg["depth_model_path"]
    depth_interval = int(depth_cfg.get("depth_interval", 1))
    is_display = bool(display_cfg.get("is_display", True))
    use_gpu = bool(prefer_cfg.get("use_gpu", False))

    # 先读一帧获取图像尺寸，用于模型初始化
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}")
        return
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        cap.release()
        return
    frame_h, frame_w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print(f"YOLO model path: {yolo_model_path}")
    print(f"Depth model path: {depth_model_path}")
    print(f"Image size: {frame_w}x{frame_h}, FPS: {fps:.1f}")
    print(f"Velocity Threshold: {motion_cfg.get('velocity_threshold')}")

    # ── 初始化 YOLO 检测器 ──
    detector = dd.YoloDetectModel()
    detector.init(
        model_path={item["type"]: item["path"] for item in yolo_model_path},
        raw_img_w=frame_w,
        raw_img_h=frame_h,
        nms_thresh=float(yolo_cfg.get("yolo_nms_thresh", 0.4)),
        conf_thresh=float(yolo_cfg.get("yolo_conf_thresh", 0.25)),
        num_class=80,
        use_gpu=use_gpu,
    )

    # ── 初始化追踪器 ──
    tracker = dd.BYTETracker(frame_rate=int(fps), track_buffer=30)

    # ── 初始化运动状态引擎 ──
    motion_engine = dd.MotionStateEngine(
        velocity_threshold=float(motion_cfg.get("velocity_threshold", 15.0)),
        acceleration_threshold=float(motion_cfg.get("acceleration_threshold", 5.0)),
        kf_process_noise_cov=float(motion_cfg.get("kf_process_noise_cov", 0.02)),
        kf_measurement_noise_cov=float(motion_cfg.get("kf_measurement_noise_cov", 0.05)),
    )

    # ── 初始化深度模型 ──
    depth_model = build_depth_model(depth_model_path, frame_w, frame_h, use_gpu)

    frame_count = 0
    latest_depth_map = None
    latest_depth_vis = None
    track_classes = { 1, 2, 3, 5,7 }
    # 处理第一帧
    frame = first_frame
    while True:
        frame_count += 1
        # ── YOLO 推理 ──
        detections = detector.inference(frame, frame_count, fps)

        # ── 构造 Object 列表 → 追踪器 ──
        objects = []
        for det in detections:
            obj = dd.Object()
            x1, y1, x2, y2 = det.bbox
            obj.rect = [x1, y1, x2 - x1, y2 - y1]
            obj.label = det.class_id
            obj.prob = det.conf
            obj.distance = 0.0
            objects.append(obj)

        tracks = tracker.update(objects)

        # ── 隔帧深度推理 ──
        if frame_count % depth_interval == 0 or latest_depth_map is None:
            latest_depth_map, latest_depth_vis = depth_model.predict(frame, frame_count, fps)

        # 用系统时间作为 motion state 的时间戳
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()

        # ── 运动状态计算 & 可视化 ──
        for track in tracks:
            raw_depth = motion_engine.get_object_depth(
                latest_depth_map, track, (frame.shape[1], frame.shape[0])
            )

            motion_state = motion_engine.compute_motion_state(
                track_id=track.track_id,
                raw_depth=raw_depth,
                timestamp=timestamp,
            )

            state_str = dd.MOTION_STR_MAP.get(
                (motion_state.state_vec, motion_state.state_acc),
                "Unknown",
            )

            draw_track_overlay(frame, track, motion_state, raw_depth, state_str)
            # print(f"Track {track.track_id}: depth={raw_depth:.2f}, state={state_str}, class={track.class_id}")

        # ── 显示 ──
        if is_display:
            show_img = frame
            if latest_depth_vis is not None:
                depth_show = latest_depth_vis
                if len(depth_show.shape) == 2:
                    depth_show = cv2.cvtColor(depth_show, cv2.COLOR_GRAY2BGR)
                depth_show = cv2.resize(depth_show, (frame_w, frame_h))
                show_img = cv2.hconcat([frame, depth_show])

            cv2.imshow("Stereo Detection", show_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        # 读取下一帧
        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()