import base64
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt

# -------------------------- MQTT 配置（需与发布端一致）--------------------------
MQTT_BROKER = (
    "fh2.wifizs.cn"  # MQTT 服务器地址（本地：localhost，远程：如 123.45.67.89）
)
MQTT_PORT = 1883  # MQTT 端口（默认 1883，SSL 加密用 8883）
MQTT_USERNAME = "wasu"  # 用户名（无则为 None）
MQTT_PASSWORD = "wasu@2025"  # 密码（无则为 None）
# CLIENT_ID = "fhDataRecerver01"  # 连接设备ID
# 订阅的主题（可多个，用列表表示）
SUBSCRIBE_TOPICS = ["thing/product/+/osd"]
QOS = 0  # 服务质量（0=最多一次，1=至少一次，2=恰好一次）
# 图片保存目录（接收 Base64 图片后保存）
IMAGE_SAVE_DIR = "../../dump"
# --------------------------------------------------------------------------------


class MQTTReceiver:
    def __init__(self):
        # 初始化 MQTT 客户端（client_id 自动生成，避免冲突）
        self.client = mqtt.Client(client_id=f"mqtt_receiver_{int(time.time())}")
        # self.client = mqtt.Client(client_id=CLIENT_ID)

        # 绑定回调函数
        self.client.on_connect = self._on_connect  # 连接成功回调
        self.client.on_message = self._on_message  # 接收消息回调
        self.client.on_disconnect = self._on_disconnect  # 断开连接回调

        # 设置用户名密码（若有）
        if MQTT_USERNAME and MQTT_PASSWORD:
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        # 创建图片保存目录
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

        # 连接状态标记
        self.connected = False

    # -------------------------- MQTT 回调函数 --------------------------
    def _on_connect(
        self, client: mqtt.Client, userdata: Any, flags: Dict[str, Any], rc: int
    ):
        """MQTT 连接成功后的回调函数"""
        if rc == 0:
            self.connected = True
            print(
                f"[{datetime.now()}] MQTT 连接成功！服务器：{MQTT_BROKER}:{MQTT_PORT}"
            )

            # 订阅指定主题（可多个）
            for topic in SUBSCRIBE_TOPICS:
                client.subscribe(topic, qos=QOS)
                print(f"[{datetime.now()}] 已订阅主题：{topic}（QOS={QOS}）")
        else:
            self.connected = False
            print(f"[{datetime.now()}] MQTT 连接失败！错误码：{rc}（rc=0 为成功）")

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int):
        """MQTT 断开连接后的回调函数"""
        self.connected = False
        if rc != 0:
            print(f"[{datetime.now()}] MQTT 意外断开连接！错误码：{rc}")
        else:
            print(f"[{datetime.now()}] MQTT 正常断开连接")

        # 自动重连（每 5 秒尝试一次）
        while not self.connected:
            print(f"[{datetime.now()}] 尝试重新连接 MQTT 服务器...")
            try:
                client.reconnect()  # 重新连接
                time.sleep(2)  # 连接成功后等待 2 秒再继续
            except Exception as e:
                print(f"[{datetime.now()}] 重连失败：{str(e)}")
                time.sleep(5)

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage):
        """接收 MQTT 消息后的回调函数"""
        try:
            # 解析消息基本信息
            topic = msg.topic
            payload = msg.payload.decode("utf-8")  # 消息内容（默认按 UTF-8 解码）
            qos = msg.qos
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            print("=" * 80)
            print(f"[{timestamp}] 收到消息：")
            print(f"  主题：{topic}")
            print(f"  QOS：{qos}")

            # 尝试解析 JSON 格式消息（若失败则按普通文本处理）
            try:
                msg_json = json.loads(payload)
                print(
                    f"  内容（JSON 格式）：{json.dumps(msg_json, ensure_ascii=False, indent=2)}"
                )

                # if "99-0-0" in msg_json["data"]["host"]:
                #     load_to_db

                # 若主题是图片，尝试解码 Base64 并保存（适配之前的 RTMP 抽帧场景）
                if (
                    topic == "rtmp/frame/image"
                    and "image_base64" in msg_json
                    and "frame_name" in msg_json
                ):
                    self._save_base64_image(
                        base64_data=msg_json["image_base64"],
                        frame_name=msg_json["frame_name"],
                    )

            except json.JSONDecodeError:
                # 非 JSON 格式，按普通文本处理
                print(f"  内容（文本格式）：{payload}")

        except Exception as e:
            print(f"[{datetime.now()}] 处理消息失败：{str(e)}")

    # -------------------------- 辅助函数：保存 Base64 图片 --------------------------
    def _save_base64_image(self, base64_data: str, frame_name: str) -> Optional[str]:
        """将 Base64 编码的图片保存为文件"""
        try:
            # Base64 解码
            image_data = base64.b64decode(base64_data)

            # 构建保存路径（保留原帧名称，避免冲突）
            image_path = os.path.join(IMAGE_SAVE_DIR, frame_name)

            # 写入文件
            with open(image_path, "wb") as f:
                f.write(image_data)

            print(f"  图片已保存：{image_path}")
            return image_path
        except Exception as e:
            print(f"  保存图片失败：{str(e)}")
            return None

    # -------------------------- 启动接收程序 --------------------------
    def run(self):
        """启动 MQTT 接收循环（阻塞模式，持续运行）"""
        print(f"[{datetime.now()}] 启动 MQTT 接收程序...")
        try:
            # 连接 MQTT 服务器（keepalive=60 表示每 60 秒发送一次心跳包）
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

            # 启动接收循环（阻塞当前线程，持续处理消息）
            self.client.loop_forever()
        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] 用户手动停止接收程序")
        except Exception as e:
            print(f"[{datetime.now()}] 启动接收程序失败：{str(e)}")
        finally:
            # 断开连接，释放资源
            self.client.disconnect()
            print(f"[{datetime.now()}] MQTT 连接已断开")


if __name__ == "__main__":
    # 创建接收实例并启动
    receiver = MQTTReceiver()
    receiver.run()
