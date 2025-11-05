use super::*;
use ffmpeg_next::ffi::AV_HWACCEL_FLAG_ALLOW_HIGH_DEPTH;
use image::{GenericImage, RgbImage};
use ort::{
    inputs,
    session::Session,
    value::{Tensor, TensorValueType, Value},
};

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

fn preprocess_image(img: &RgbImage, target_size: usize) -> Result<Value<TensorValueType<f32>>> {
    let resized_img = image::imageops::resize(
        img,
        target_size as u32,
        target_size as u32,
        image::imageops::FilterType::Lanczos3,
    );
    const BATCH_SIZE: usize = 1;
    const CHANNELS: usize = 3;
    let data_len = BATCH_SIZE * CHANNELS * target_size as usize * target_size as usize;
    let mut input_data = vec![0.0; data_len];
    for i in 0..BATCH_SIZE {
        for j in 0..CHANNELS {
            for k_ in 0..target_size {
                for l_ in 0..target_size {
                    let k = k_ as usize;
                    let l = l_ as usize;
                    let idx = i * CHANNELS * target_size * target_size
                        + j * target_size * target_size
                        + k * target_size
                        + l;
                    // data[idx] = resized_img[(k, l)][j] as f32 / 255.0;
                    input_data[idx] = resized_img.get_pixel(l as u32, k as u32).0[j] as f32;
                }
            }
        }
    }
    let input_shape = [BATCH_SIZE, CHANNELS, target_size, target_size];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))?;
    Ok(input_tensor)
}

fn predict_img(
    session: &mut Session,
    input_tensor: Value<TensorValueType<f32>>,
    labels: &[String],
    conf: f32,
    img: &RgbImage,
    target_size: usize,
) -> Result<Vec<(BoundingBox, String, f32)>> {
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
        let label = labels[class_id].clone();
        let xc = row[0] / target_size_f32 * (w as f32);
        let yc = row[1] / target_size_f32 * (h as f32);
        let w = row[2] / target_size_f32 * (w as f32);
        let h = row[3] / target_size_f32 * (h as f32);
        let b = BoundingBox {
            x1: xc - w / 2.,
            y1: yc - h / 2.,
            x2: xc + w / 2.,
            y2: yc + h / 2.,
        };
        boxes.push((b, label, prob));
        // println!("class_id: {}, prob: {}, label: {}, boxes: {:?}", class_id, prob, label, boxes);
    }
    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));

    Ok(boxes)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_div_predict() -> Result<()> {
        let mut session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .commit_from_file("./model/yolo11n_visdrone.onnx")?;
        let meta = session.metadata()?;
        println!("meta keys: {:?}", meta.custom_keys());
        // 显然是直接传入labels比较正确且方便
        //meta: Ok(Some("{0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}"))
        let names = meta.custom("names")?.unwrap_or("{}".into());
        println!("names: {}", &names);
        let labels: HashMap<i32, String> = match serde_json::from_str(&names) {
            Ok(labels) => labels,
            Err(e) => {
                eprintln!("Failed to parse labels: {}", e);
                HashMap::new()
            }
        };

        println!("laebels: {:?}", &labels);

        for (idx, inf) in session.inputs.iter().enumerate() {
            // name: "images"
            println!("id: {}: {}", idx, inf.name);
        }
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
        assert!(true);
        Ok(())
    }
}
