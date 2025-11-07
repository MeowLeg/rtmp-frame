import os
import sqlite3
from dataclasses import dataclass
from time import sleep

import cv2
import numpy as np
import requests
from cv2.typing import MatLike
from ultralytics.engine.results import Results
from ultralytics.models.yolo.model import YOLO

# 实现的模型
MODELS = {
    "swim_reservoir": "../model/yolo11n_visdrone.pt",  # 可以有多种类型的判断
    "fire_forest": "../model/fire_tolo8_small.pt",
}


def split_image(
    image: MatLike, n: int = 2
) -> tuple[list[MatLike], list[tuple[int, int]]]:
    # 分割图片
    shape: tuple[int, int, int] = image.shape  # pyright: ignore[reportAny]
    h, w = shape[:2]
    sub_h, sub_w = h // n, w // n
    sub_images: list[MatLike] = []
    positions: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            y1 = i * sub_h
            y2 = (i + 1) * sub_h if i < n - 1 else h
            x1 = j * sub_w
            x2 = (j + 1) * sub_w if j < n - 1 else w
            sub_img = image[y1:y2, x1:x2]
            sub_images.append(sub_img)
            positions.append((x1, y1))
    return sub_images, positions


def detect_split_image(
    model: YOLO, image: MatLike, conf: float = 0.5, iou: float = 0.45
):
    # 对分割的图片进行检测
    sub_images, positions = split_image(image, n=2)
    all_boxes: list[list[float | int]] = []

    for sub_img, (x_offset, y_offset) in zip(sub_images, positions):
        results: list[Results] = model(sub_img, conf=conf, iou=iou)  # pyright: ignore[reportUnknownVariableType]
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            # xyxy: tensor([[730.1935, 276.2485, 804.1757, 315.5241]])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # pyright: ignore[reportAny, reportUnknownMemberType]
            x1 += x_offset  # pyright: ignore[reportAny]
            y1 += y_offset  # pyright: ignore[reportAny]
            x2 += x_offset  # pyright: ignore[reportAny]
            y2 += y_offset  # pyright: ignore[reportAny]

            cls: int = int(box.cls[0])  # pyright: ignore[reportUnknownMemberType]
            box_conf: float = float(box.conf[0])  # pyright: ignore[reportUnknownMemberType]
            all_boxes.append([x1, y1, x2, y2, box_conf, cls])

    if not all_boxes:
        return np.array([])
    n_all_boxes = np.array(all_boxes)
    boxes = n_all_boxes[:, :4]
    scores = n_all_boxes[:, 4]
    nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou)  # pyright: ignore[reportAny]
    if len(nms_indices) == 0:
        return np.array([])
    indices = (  # pyright: ignore[reportUnknownVariableType]
        nms_indices.flatten() if isinstance(nms_indices, np.ndarray) else nms_indices[0]
    )
    return n_all_boxes[indices]  # pyright: ignore[reportAny]

    # yolo-11
    # pred = torch.tensor(all_boxes[:, :4], dtype=torch.float32),  # 框
    # pred = (torch.tensor(all_boxes[:, :4]), torch.tensor(all_boxes[:, 4]), torch.tensor(all_boxes[:, 5]))
    # # 使用 YOLO 内置 NMS
    # nms_results = non_max_suppression(
    #     prediction=pred,
    #     conf_thres=conf,
    #     iou_thres=iou
    # )[0]  # 返回 tensor
    # return nms_results.cpu().numpy()


def predict(
    model: YOLO, out_path: str, im_path: str, im_name: str, conf: float
) -> list[str]:
    image: MatLike | None = cv2.imread(im_path)
    if image is None:
        return []

    merged_boxes = detect_split_image(model, image, conf=conf, iou=0.45)

    labels_result: list[str] = []
    for box in merged_boxes:  # pyright: ignore[reportAny]
        # float, float, float, float, float, int
        x1, y1, x2, y2, conf, cls = box  # pyright: ignore[reportAny]
        _ = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # pyright: ignore[reportAny]
        label = f"{model.names[int(cls)]} {conf: .2f}"  # pyright: ignore[reportAny]
        labels_result.append(model.names[int(cls)])  # pyright: ignore[reportAny]
        y1_text = int(y1) - 10  # pyright: ignore[reportAny]
        if y1_text <= 0:
            y1_text = 0
        _ = cv2.putText(
            image,
            label,
            (int(x1), y1_text),  # pyright: ignore[reportAny]
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        _ = cv2.imwrite(os.path.join(out_path, im_name), image)

    return labels_result


@dataclass
class Image:
    id: str
    path: str
    uuid: str
    create_date: str


@dataclass
class Stream:
    uuid: str
    tags: list[str]
    images: list[Image]


def read_from_db(db_path: str) -> list[Stream]:
    # 从数据库读取图片进行检测
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    streams: list[Stream] = []
    rs: list[sqlite3.Row] = cur.execute("""
        select uuid from stream where is_over = 0
    """).fetchall()
    for r_ in rs:
        r: dict[str, str] = dict(r_)
        tags: list[sqlite3.Row] = cur.execute(
            "select code from stream_tag where uuid = ?", (r["uuid"],)
        ).fetchall()
        images_: list[sqlite3.Row] = cur.execute(
            "select id, path, uuid, create_date from pic where predicted = 0 and uuid = ?",
            (r["uuid"],),
        ).fetchall()
        images: list[dict[str, str]] = [dict(i) for i in images_]
        stream: Stream = Stream(
            r["uuid"],
            [r_["code"] for r_ in tags],
            [Image(**dict(r_)) for r_ in images],
        )
        streams.append(stream)
    return streams


def filter_models(tags: list[str]) -> list[YOLO]:
    # 列出实现的模型
    # 类似SET
    f_models: dict[str, None] = {}
    for tag in tags:
        if MODELS[tag]:
            f_models[MODELS[tag]] = None
    return [YOLO(m) for m in list(f_models.keys())]


@dataclass
class Alert:
    uuid: str
    title: str
    content: str
    datetime: str
    alert_type: list[str]
    mediaUrl: str
    videoUrl: list[str]
    otherUrl: list[str]


def loop_predict(
    output_dir: str,
    db_path: str,
    interval: int,
    conf: float = 0.5,
) -> None:
    # 循环检测
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    while True:
        im_predicted: list[Alert] = []
        for stream in read_from_db(db_path):
            for im in stream.images:
                im_name = os.path.basename(im.path)
                for m in filter_models(stream.tags):
                    labels = predict(
                        m,
                        output_dir,
                        im.path,
                        im_name,
                        conf,
                    )
                    if len(labels):
                        im_predicted.append(
                            Alert(
                                im.uuid,
                                "",
                                "",
                                im.create_date,
                                labels,
                                im.path,
                                [],
                                [],
                            )
                        )
            _ = cur.execute(
                "update pic set predicted = 1 where id in ("
                + ",".join([str(im.id) for im in stream.images])
                + ")"
            )
            db.commit()
        # todo: 通知远程服务器，todo里的yoloAlert.json
        for p in im_predicted:
            # 测试环境
            svr_url = "http://192.168.213.226:11005/v1/webhook/yoloalert"
            # 正式环境
            # https://fh2.wifizs.cn/11005/v1/webhook/yoloalert
            response = requests.post(svr_url, json=p)
            print(response.text)
        # 等待
        sleep(interval)


if __name__ == "__main__":
    # model = YOLO("../model/yolo11n_visdrone.pt")
    # input_dir = "../dump"
    # output_dir = "../static"
    # for im in os.listdir(input_dir):
    #     if im.lower().endswith((".jpg", ".png", ".jpeg")):
    #         im_path = os.path.join(input_dir, im)
    #         _ = predict(model, output_dir, im_path, im, 0.5)

    # loop_predict("../dump/", "../static/", "../predict.db", 30, 0.5)

    # im = cv2.imread("../dump/f353e1b849e5f8f5e6b740359f0c5858_20251029174000_2280.jpg")
    # im = cv2.imread("../dump/f353e1b849e5f8f5e6b740359f0c5858_20251029174126_4440.jpg")
    # rslt = split_image(im, 2)
    # print(rslt)

    predict(
        YOLO("../model/yolo11n_visdrone.pt"),
        "../static/",
        "../dump/f353e1b849e5f8f5e6b740359f0c5858_20251029174000_2280.jpg",
        "f353e1b849e5f8f5e6b740359f0c5858_20251029174000_2280.jpg",
        0.5,
    )
