from http.client import CREATED
import os
import cv2
import sqlite3
from cv2.typing import MatLike
from pandas.core.apply import ResType
import requests
from time import sleep
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from dataclasses import dataclass


# 实现的模型
MODELS = {
    "swim_reservoir": "../model/yolo11n_visdrone.pt",  # 可以有多种类型的判断
    "fire_forest": "../model/fire_tolo8_small.pt",
}


def split_image(image: MatLike, n: int = 2):
    # 分割图片
    h, w = image.shape[:2]
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
    all_boxes = []

    for sub_img, (x_offset, y_offset) in zip(sub_images, positions):
        results: list[Results] = model(sub_img, conf=conf, iou=iou)
        boxes = results[0].boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 += x_offset
            y1 += y_offset
            x2 += x_offset
            y2 += y_offset

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            all_boxes.append([x1, y1, x2, y2, conf, cls])

    if not all_boxes:
        return np.array([])
    all_boxes = np.array(all_boxes)
    boxes = all_boxes[:, :4]
    scores = all_boxes[:, 4]
    nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou)
    if len(nms_indices) == 0:
        return np.array([])
    indices = (
        nms_indices.flatten() if isinstance(nms_indices, np.ndarray) else nms_indices[0]
    )
    return all_boxes[indices]

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
    model: YOLO, in_path: str, out_path: str, im_path: str, im_name: str, conf: float
) -> list[str]:
    image: MatLike = cv2.imread(im_path)
    merged_boxes = detect_split_image(model, image, conf=conf, iou=0.45)

    labels_result = []
    for box in merged_boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{model.names[int(cls)]} {conf: .2f}"
        labels_result.append(model.names[int(cls)])
        y1_text = int(y1) - 10
        if y1_text <= 0:
            y1_text = 0
        _ = cv2.putText(
            image,
            label,
            (int(x1), y1_text),
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


def loop_predict(
    input_dir: str,
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
        im_predicted = []
        for stream in read_from_db(db_path):
            for im in stream.images:
                im_name = os.path.basename(im.path)
                for m in filter_models(stream.tags):
                    labels = predict(
                        m,
                        input_dir,
                        output_dir,
                        im.path,
                        im_name,
                        conf,
                    )
                    if len(labels):
                        im_predicted.append(
                            {
                                "uuid": im.uuid,
                                "title": "",
                                "content": "",
                                "datetime": im.create_date,
                                "alert_type": labels,
                                "mediaUrl": im.path,
                                "videoUrl": [],
                                "otherUrl": [],
                            }
                        )
            _ = cur.execute(
                "update pic set predicted = 1 where id in ("
                + ",".join([str(im["id"]) for im in stream.images])
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
            print(response.json())
        # 等待
        sleep(interval)


if __name__ == "__main__":
    model = YOLO("../model/yolo11n_visdrone.pt")
    input_dir = "../dump"
    output_dir = "../static"
    for im in os.listdir(input_dir):
        if im.lower().endswith((".jpg", ".png", ".jpeg")):
            im_path = os.path.join(input_dir, im)
            predict(model, input_dir, output_dir, im_path, im, 0.5)

    # loop_predict("../dump/", "../static/", "../predict.db", 30, 0.5)
