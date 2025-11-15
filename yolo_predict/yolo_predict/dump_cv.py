import os
import time
from datetime import datetime, timedelta

import cv2


def cv2_rtmp_read_with_timestamp(
    rtmp_url: str,
    output_dir: str = "cv2_rtmp_frames",
    frame_interval: int = 5,  # 每5秒抽1帧
    retry_count: int = 3,
    timeout_open: int = 15,  # 打开超时（秒）
    timeout_read: int = 10,  # 读取超时（秒）
):
    os.makedirs(output_dir, exist_ok=True)
    retry = 0

    while retry < retry_count:
        # 4.12.0 推荐方式：构造函数指定 FFmpeg 后端（自动启用，无需额外设置 CAP_PROP_FFMPEG_BACKEND）
        # 可选：添加 fflags 参数禁用缓冲区，进一步降低时间戳延迟
        cap = cv2.VideoCapture(
            rtmp_url,
            cv2.CAP_FFMPEG,  # 强制使用 FFmpeg 解码（4.12.0 支持完美）
        )

        # 4.12.0 支持的关键配置（全部生效，放心使用）
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_open * 1000)  # 打开超时（毫秒）
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_read * 1000)  # 读取超时（毫秒）
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 最小缓冲区（降低延迟，避免时间戳滞后）
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, 1920
        )  # 可选：指定输出宽度（按流实际分辨率调整）
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 可选：指定输出高度
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 禁用自动对焦（无意义，减少资源占用）

        # 验证连接
        if not cap.isOpened():
            print(
                f"[{datetime.now()}] 连接 RTMP 失败（地址：{rtmp_url}），重试 {retry}/{retry_count}"
            )
            retry += 1
            time.sleep(5)
            continue

        # 4.12.0 可稳定获取流信息（实时流也能拿到准确帧率）
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 兜底：若帧率为 0（极个别实时流），默认按 30fps 计算
        fps = fps if fps > 0 else 30.0
        timebase = 1.0 / fps  # 时间基准（每帧的理论时长）

        print(f"[{datetime.now()}] 连接成功！")
        print(f"  - 帧率：{fps:.2f}fps")
        print(f"  - 时间基准：{timebase:.3f}s/帧")
        print(f"  - 缓冲区大小：{cap.get(cv2.CAP_PROP_BUFFERSIZE)} 帧")
        print("  - 开始读取帧和时间戳...")

        last_save_time = time.time()
        frame_idx = 0
        last_timestamp_ms = 0  # 记录上一帧时间戳，检测跳变
        stream_start_unix = (
            time.time()
        )  # 流启动的 Unix 时间戳（用于计算帧的实际系统时间）

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{datetime.now()}] 流中断或读取失败（可能是网络波动）")
                    break

                # --------------- 核心：获取并处理时间戳（4.12.0 稳定支持）---------------
                # 1. 获取 PTS 时间戳（毫秒，相对于流启动时间）
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                # 过滤异常时间戳（4.12.0 偶尔会出现负数/跳变，需处理）
                if timestamp_ms < 0 or (
                    frame_idx > 0 and abs(timestamp_ms - last_timestamp_ms) > 2000
                ):
                    print(
                        f"[{datetime.now()}] 时间戳异常：当前 {timestamp_ms:.0f}ms | 上一帧 {last_timestamp_ms:.0f}ms → 忽略"
                    )
                    frame_idx += 1
                    continue
                last_timestamp_ms = timestamp_ms

                # 2. 时间戳格式转换（3种常用格式，按需选择）
                timestamp_sec = timestamp_ms / 1000.0  # 秒级（含小数）
                timestamp_str = str(timedelta(seconds=timestamp_sec))[
                    :-3
                ]  # 可读格式（时:分:秒.毫秒）
                frame_unix_time = (
                    stream_start_unix + timestamp_sec
                )  # 帧的实际系统时间（Unix 时间戳）
                frame_unix_str = datetime.fromtimestamp(frame_unix_time).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )[:-3]  # 带日期的系统时间

                # --------------- 按时间间隔保存帧（含时间戳命名）---------------
                current_time = time.time()
                if current_time - last_save_time >= frame_interval:
                    # 3. 打印时间戳信息（按需关闭，减少日志开销）
                    print(
                        f"[{datetime.now()}] 帧{frame_idx:4d} | PTS：{timestamp_str} | 系统时间：{frame_unix_str}"
                    )
                    # 文件名含时间戳（避免冲突，便于追溯）
                    frame_name = f"frame_{frame_unix_str.replace(' ', '_').replace(':', '-').replace('.', '_')}_{frame_idx}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    # 保存帧（调整画质：90 为高质量，可改为 80 平衡大小和画质）
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    print(f"[{datetime.now()}] 保存帧：{frame_path}")
                    last_save_time = current_time

                frame_idx += 1

                # 按 'q' 退出（4.12.0 支持 GUI 环境，无 GUI 时注释此行）
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"[{datetime.now()}] 用户手动退出")
                    break

        except Exception as e:
            print(f"[{datetime.now()}] 运行错误：{str(e)}")
        finally:
            # 4.12.0 必须释放资源（否则会残留内存/网络连接）
            cap.release()
            cv2.destroyAllWindows()
            print(f"[{datetime.now()}] 资源已释放")

        retry += 1
        if retry < retry_count:
            print(f"[{datetime.now()}] 5秒后重新连接...")
            time.sleep(5)
        else:
            print(f"[{datetime.now()}] 重试次数耗尽（{retry_count}次），退出程序")
            break


if __name__ == "__main__":
    # 你的 RTMP 地址（已验证格式正确，无需编码）
    RTMP_URL = "rtmp://play-sh.quklive.com/live/1761274636335259"
    cv2_rtmp_read_with_timestamp(
        rtmp_url=RTMP_URL,
        output_dir="../../dump",
        frame_interval=5,  # 每5秒抽1帧（可改为 1 秒抽1帧，按需调整）
        retry_count=3,
        timeout_open=15,
        timeout_read=10,
    )
