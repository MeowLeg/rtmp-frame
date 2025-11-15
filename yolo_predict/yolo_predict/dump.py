import os
import time
from datetime import datetime

import ffmpeg


def rtmp_frame_extract(
    rtmp_url: str,
    output_dir: str = "extracted_frames",
    frame_interval: int = 10,  # 每10秒抽1帧
    retry_count: int = 5,  # 断线重试次数
    timeout: int = 15,  # 流超时时间（秒）
):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    retry = 0
    while retry < retry_count:
        try:
            print(f"[{datetime.now()}] 连接 RTMP 流：{rtmp_url}")

            # FFmpeg 命令参数（核心逻辑）
            (
                ffmpeg.input(
                    rtmp_url,
                    # rtsp_transport="tcp",  # RTMP 用 TCP 传输，减少丢包
                    # timeout=timeout,       # 超时时间
                    # stimeout=timeout * 1000  # 网络超时（毫秒）Linux 也许可用
                )
                .filter("fps", fps=f"1/{frame_interval}")  # 按时间间隔抽帧（1帧/10秒）
                .output(
                    os.path.join(output_dir, "frame_%Y%m%d_%H%M%S.jpg"),  # 时间戳命名
                    vcodec="mjpeg",  # 输出 JPEG 格式
                    qscale=2,  # 画质（1-31，越小越清晰）
                    strftime=True,  # 启用时间戳格式化文件名
                )
                .overwrite_output()  # 覆盖同名文件（可选）
                .run(capture_stdout=True, capture_stderr=True)  # 捕获日志
            )

            # 正常退出（流结束）
            print(f"[{datetime.now()}] 流正常结束")
            break

        except ffmpeg.Error as e:
            # 捕获 FFmpeg 错误日志
            stderr = e.stderr.decode("utf-8", errors="ignore")
            print(
                f"[{datetime.now()}] 抽帧失败（重试 {retry}/{retry_count}）：{stderr}"
            )

            retry += 1
            if retry < retry_count:
                print(f"[{datetime.now()}] 5秒后重新连接...")
                time.sleep(5)
            else:
                print(f"[{datetime.now()}] 重试次数耗尽，退出")
                raise
        except Exception as e:
            print(f"[{datetime.now()}] 未知错误：{str(e)}")
            retry += 1
            time.sleep(5)


if __name__ == "__main__":
    # 替换为你的 RTMP 流地址（如 rtmp://live.hkstv.hk.lxdns.com/live/hks）
    RTMP_URL = "rtmp://play-sh.quklive.com/live/1761274636335259"
    rtmp_frame_extract(
        rtmp_url=RTMP_URL,
        output_dir="../../dump",
        frame_interval=6,  # 每5秒抽1帧
    )
