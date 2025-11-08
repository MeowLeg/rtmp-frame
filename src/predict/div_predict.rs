use super::*;
// use ffmpeg_next::{ffi::AV_LOG_PRINT_LEVEL, software::scaling::support::output};
use image::{GenericImage, Rgb, RgbImage};
use ort::{
    inputs,
    session::Session,
    value::{Tensor, TensorValueType, Value},
};
use std::fs::read_dir;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: u32,
}

fn split_image(image: &mut RgbImage, n: u32) -> Result<(Vec<RgbImage>, Vec<(u32, u32)>)> {
    let (orig_w, orig_h) = image.dimensions();

    let sub_w = orig_w / n;
    let sub_h = orig_h / n;

    let mut sub_rgb_images: Vec<RgbImage> = Vec::with_capacity((n * n) as usize);
    let mut positions: Vec<(u32, u32)> = Vec::with_capacity((n * n) as usize);

    for i in 0..n {
        for j in 0..n {
            let x = j * sub_w;
            let y = i * sub_h;
            let w = if j < n - 1 { sub_w } else { orig_w - x };
            let h = if i < n - 1 { sub_h } else { orig_h - y };
            // println!("x: {}, y: {}, w: {}, h: {}", x, y, w, h);
            let sub_rgb_image = image.sub_image(x, y, w, h).to_image();
            sub_rgb_images.push(sub_rgb_image);
            positions.push((x, y));
        }
    }

    Ok((sub_rgb_images, positions))
}

fn letterbox_resize(image: &RgbImage, target_size: usize) -> Result<RgbImage> {
    let (orig_w, orig_h) = image.dimensions();
    // println!("Original dimensions: {}x{}", orig_w, orig_h);

    let scale = target_size as f32 / orig_w.max(orig_h) as f32;
    // println!("Scale: {}", scale);

    let scaled_w = (orig_w as f32 * scale).round();
    let scaled_h = (orig_h as f32 * scale).round();
    // println!("Scaled dimensions: {}x{}", scaled_w, scaled_h);

    let padding_w = (target_size as f32 - scaled_w) / 2.0;
    let padding_h = (target_size as f32 - scaled_h) / 2.0;

    let resized_image = image::imageops::resize(
        image,
        scaled_w as u32,
        scaled_h as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let mut padded_image = ImageBuffer::from_pixel(
        target_size as u32,
        target_size as u32,
        Rgb([0, 0, 0]), // 黑边（RGB: 0,0,0）
    );

    image::imageops::overlay(
        &mut padded_image,
        &resized_image,
        padding_w as i64, // X轴偏移（左填充量）
        padding_h as i64, // Y轴偏移（上填充量）
    );

    Ok(padded_image)
}

fn preprocess_image(img: &RgbImage, target_size: usize) -> Result<Value<TensorValueType<f32>>> {
    // 不能使用resize
    // let resized_img = image::imageops::resize(
    //     img,
    //     target_size as u32,
    //     target_size as u32,
    //     image::imageops::FilterType::Lanczos3,
    // );
    let resized_img = letterbox_resize(img, target_size)?;
    // println!("dimensions after resize: {:?}", resized_img.dimensions());
    const BATCH_SIZE: usize = 1;
    const CHANNELS: usize = 3;
    let data_len = BATCH_SIZE * CHANNELS * target_size as usize * target_size as usize;
    let mut input_data = vec![0.0; data_len];
    for i in 0..BATCH_SIZE {
        for j in 0..CHANNELS {
            for k in 0..target_size {
                for l in 0..target_size {
                    let idx = i * CHANNELS * target_size * target_size
                        + j * target_size * target_size
                        + k * target_size
                        + l;
                    input_data[idx] =
                        (resized_img.get_pixel(l as u32, k as u32).0[j] as f32) / 255.0;
                }
            }
        }
    }
    let input_shape = [BATCH_SIZE, CHANNELS, target_size, target_size];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))?;
    Ok(input_tensor)
}

fn predict_image(
    session: &mut Session,
    conf: f32,
    img: &RgbImage,
    target_size: usize,
    input_tensor: Value<TensorValueType<f32>>,
    position: (u32, u32),
    // ) -> Result<Vec<(BoundingBox, u32, f32)>> {
) -> Result<Vec<BBox>> {
    let (w, h) = img.dimensions();
    let outputs = session.run(inputs![input_tensor])?;
    let output = outputs["output0"]
        .try_extract_array::<f32>()?
        .t()
        .into_owned();
    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);

    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        // conf: 0.5
        if prob < conf {
            continue;
        }
        let target_size_f32 = target_size as f32;
        let xc = row[0] / target_size_f32 * (w as f32);
        let yc = row[1] / target_size_f32 * (h as f32);
        let w = row[2] / target_size_f32 * (w as f32);
        let h = row[3] / target_size_f32 * (h as f32);
        boxes.push(BBox {
            x1: xc - w / 2. + (position.0 as f32),
            y1: yc - h / 2. + (position.1 as f32),
            x2: xc + w / 2. + (position.0 as f32),
            y2: yc + h / 2. + (position.1 as f32),
            score: prob,
            class_id: class_id as u32,
        });
    }
    // boxes.sort_by(|box1, box2| box2.score.total_cmp(&box1.score));

    Ok(boxes)
}

fn merge_boexs(bboxes_vec: Vec<Vec<BBox>>, iou_threshold: f32) -> Vec<BBox> {
    let mut all_bboxes: Vec<BBox> = Vec::new();
    for bboxes in bboxes_vec.into_iter() {
        all_bboxes.extend(bboxes);
    }
    all_bboxes.sort_by(|box1, box2| box2.score.total_cmp(&box1.score));
    let final_bboxes = nms(&all_bboxes, iou_threshold);
    final_bboxes
}

fn nms(sorted_bboxes: &[BBox], iou_threshold: f32) -> Vec<BBox> {
    if sorted_bboxes.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();
    let mut removed = vec![false; sorted_bboxes.len()];

    for i in 0..sorted_bboxes.len() {
        if removed[i] {
            continue;
        }

        let current = sorted_bboxes[i];
        result.push(current);

        for j in (i + 1)..sorted_bboxes.len() {
            if removed[j] {
                continue;
            }

            let other = sorted_bboxes[j];
            let iou = calculate_iou(&current, &other);

            if iou >= iou_threshold {
                removed[j] = true;
            }
        }
    }

    result
}

fn calculate_iou(a: &BBox, b: &BBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let inter_area = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));
    let a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
    let b_area = (b.x2 - b.x1) * (b.y2 - b.y1);

    inter_area / (a_area + b_area - inter_area)
}

fn visualize_detections(
    image_path: &PathBuf,
    bboxes: Vec<BBox>,
    out_pf: &PathBuf,
    labels: &[String],
) -> Result<()> {
    // Load the original image
    let mut img = image::open(image_path)?.to_rgb8();

    let font = FontRef::try_from_slice(include_bytes!("../../font/Deng.ttf"))?;
    let scale = PxScale {
        x: 20.0_f32,
        y: 20.0_f32,
    };

    for bbox in bboxes.into_iter() {
        // Draw bounding box
        let rect = imageproc::rect::Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);
        draw_hollow_rect_mut(&mut img, rect, Rgb([255, 0, 0]));

        // Draw label and probability
        let text = format!("{} ({:.2})", labels[bbox.class_id as usize], bbox.score);
        draw_text_mut(
            &mut img,
            Rgb([255, 255, 255]),
            bbox.x1 as i32,
            bbox.y1 as i32 - 20,
            scale,
            &font,
            &text,
        );
    }
    img.save(out_pf)?;

    Ok(())
}

fn predict(
    // model_path: &str,
    session: &mut Session,
    img_path: &str,
    out_dir: &str,
    conf: f32,
    iou: f32,
    n: u32,
    target_size: usize,
    labels: &[String],
) -> Result<()> {
    let mut image: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(img_path)?.into_rgb8();

    let (sub_rgb_images, positions) = split_image(&mut image, n)?;

    let mut bboxes_vec = Vec::new();
    for (sub_image, position) in sub_rgb_images.into_iter().zip(positions) {
        let input_tensor_data = preprocess_image(&sub_image, target_size)?;
        let bboxes = predict_image(
            session,
            conf,
            &sub_image,
            target_size,
            input_tensor_data,
            position,
        )?;
        bboxes_vec.push(bboxes);
    }
    let in_image = PathBuf::from(img_path);
    let in_image_name = in_image.file_name().unwrap().to_str().unwrap();
    let out_image = PathBuf::from(out_dir).join(in_image_name);
    let out_ordered_boxes = merge_boexs(bboxes_vec, iou);
    println!("out_ordered_boxes: {:?}", &out_ordered_boxes);
    if !out_ordered_boxes.is_empty() {
        let _ = visualize_detections(&in_image, out_ordered_boxes, &out_image, &labels)?;
    }

    Ok(())
}

pub fn predict_dir(
    model_path: &str,
    in_dir: &str,
    out_dir: &str,
    conf: f32,
    iou: f32,
    labels: &[String],
    n: u32,
) -> Result<()> {
    // let model_path = "./model/yolo11n_visdrone.onnx";
    let mut session = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(model_path)?;
    for entry in read_dir(Path::new(in_dir))? {
        let entry = entry?;
        let ep = entry.path();
        if ep.is_file() {
            let im_path = ep.to_str().unwrap();
            // println!("im_path: {}", im_path);
            let _ = predict(
                &mut session,
                im_path,
                out_dir,
                // 0.5,
                // 0.7,
                conf,
                iou,
                n,
                640,
                labels
                // vec![
                //     "pedestrian",
                //     "people",
                //     "bicycle",
                //     "car",
                //     "van",
                //     "truck",
                //     "tricycle",
                //     "awning-tricycle",
                //     "bus",
                //     "motor",
                // ],
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::map::Entry;

    use super::*;
    use std::fs::*;
    use std::path::Path;

    #[test]
    fn test_div_predict() -> Result<()> {
        // let mut session = Session::builder()?
        //     .with_execution_providers([CUDAExecutionProvider::default().build()])?
        //     .commit_from_file("./model/yolo11n_visdrone.onnx")?;
        // let meta = session.metadata()?;
        // println!("meta keys: {:?}", meta.custom_keys());
        // // 显然是直接传入labels比较正确且方便
        // //meta: Ok(Some("{0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}"))
        // let names = meta.custom("names")?.unwrap_or("{}".into());
        // println!("names: {}", &names);
        // let labels: HashMap<i32, String> = match serde_json::from_str(&names) {
        //     Ok(labels) => labels,
        //     Err(e) => {
        //         eprintln!("Failed to parse labels: {}", e);
        //         HashMap::new()
        //     }
        // };

        // println!("laebels: {:?}", &labels);

        // for (idx, inf) in session.inputs.iter().enumerate() {
        //     // name: "images"
        //     println!("id: {}: {}", idx, inf.name);
        // }
        // // let start_1 = Instant::now();
        // let mut image: ImageBuffer<Rgb<u8>, Vec<u8>> =
        //     image::open("./dump/f353e1b849e5f8f5e6b740359f0c5858_20251029173833_120.jpg")?
        //         .into_rgb8();
        // // println!("open image use {} ms", start_1.elapsed().as_millis());
        // // println!("Image dimensions: {}x{}", image.width(), image.height());
        // let n = 2;
        // let (images, positions) = split_image(&mut image, n)?;
        // // println!("split image use {} ms", start_1.elapsed().as_millis());
        // let target_size = 640;
        // for (idx, (sub_img, (x, y))) in images.into_iter().zip(positions).into_iter().enumerate() {
        //     let input_tensor = preprocess_image(&sub_img, target_size)?;
        //     let box = predict_img(&mut session, input_tensor, labels, conf, img, target_size)
        // }

        let model_path = "./model/yolo11n_visdrone.onnx";
        let mut session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .commit_from_file(model_path)?;
        for entry in read_dir(Path::new("./dump"))? {
            let entry = entry?;
            let ep = entry.path();
            if ep.is_file() {
                let im_path = ep.to_str().unwrap();
                println!("im_path: {}", im_path);
                let _ = predict(
                    &mut session,
                    im_path,
                    "./static/",
                    0.5,
                    0.7,
                    2,
                    640,
                    &vec![
                        "pedestrian".to_string(),
                        "people".to_string(),
                        "bicycle".to_string(),
                        "car".to_string(),
                        "van".to_string(),
                        "truck".to_string(),
                        "tricycle".to_string(),
                        "awning-tricycle".to_string(),
                        "bus".to_string(),
                        "motor".to_string(),
                    ],
                )?;
            }
        }
        assert!(true);
        Ok(())
    }
}
