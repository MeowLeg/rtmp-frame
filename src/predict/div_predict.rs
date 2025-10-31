use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use ndarray::{Array, IxDyn};
use ort::{GraphOptimizationLevel, Session, Tensor, execution_providers::CudaExecutionProvider};
use std::path::Path;

// 分块配置
const PATCH_SIZE: u32 = 512;
const OVERLAP: u32 = 64;

pub fn div_predict() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 直接构建会话（无需 Environment）
    let mut session_builder =
        Session::builder()? // 移除 Environment，直接调用 Session::builder()
            .with_graph_optimization_level(GraphOptimizationLevel::Level3)?;

    // 可选：启用 CUDA 加速
    #[cfg(feature = "cuda")]
    {
        session_builder =
            session_builder.with_execution_providers([CudaExecutionProvider::default()])?;
    }

    // 加载模型
    let session = session_builder.with_model_from_file("segmentation_model.onnx")?;

    // 2. 加载原图
    let original_image = image::open(Path::new("large_image.jpg"))?;
    let (orig_width, orig_height) = original_image.dimensions();
    println!("原图尺寸: {}x{}", orig_width, orig_height);

    // 3. 生成子块坐标
    let patches = generate_patch_coordinates(orig_width, orig_height);
    println!("子块数量: {}", patches.len());

    // 4. 初始化全局结果
    let mut global_mask = vec![0u8; (orig_width * orig_height) as usize];
    let mut pixel_counts = vec![0u32; (orig_width * orig_height) as usize];

    // 5. 处理子块（后续逻辑不变）
    for (idx, (x, y, w, h)) in patches.iter().enumerate() {
        println!("处理子块 {}: ({}, {}), {}x{}", idx, x, y, w, h);

        // a. 裁剪子块
        let patch = original_image.crop(*x, *y, *w, *h);

        // b. 预处理
        let preprocessed = preprocess_patch(&patch, PATCH_SIZE as usize);

        // c. 准备输入张量
        let input_array = Array::from_shape_vec(
            IxDyn(&[1, 3, PATCH_SIZE as usize, PATCH_SIZE as usize]),
            preprocessed,
        )?;
        let input_tensor = Tensor::from_array(session.allocator(), &input_array)?;

        // d. 推理
        let outputs = session.run([("input", input_tensor)])?;

        // e. 解析输出
        let output_tensor = outputs
            .into_iter()
            .find(|(name, _)| name == "output")
            .expect("未找到输出")
            .1
            .into_tensor::<f32>()?;
        let output_array: Array<f32, IxDyn> = output_tensor.try_into()?;

        // f. 后处理子块
        let patch_mask = postprocess_patch(&output_array, *w, *h);

        // g. 合并结果
        merge_patch_mask(
            &patch_mask,
            *x,
            *y,
            *w,
            *h,
            orig_width,
            &mut global_mask,
            &mut pixel_counts,
        );
    }

    // 6. 保存结果
    save_final_mask(&global_mask, orig_width, orig_height, "result.png")?;

    Ok(())
}

// 辅助函数（与之前一致）
fn generate_patch_coordinates(orig_w: u32, orig_h: u32) -> Vec<(u32, u32, u32, u32)> {
    let step = PATCH_SIZE - OVERLAP;
    let mut patches = Vec::new();
    let mut y = 0;
    while y < orig_h {
        let mut x = 0;
        while x < orig_w {
            let w = std::cmp::min(PATCH_SIZE, orig_w - x);
            let h = std::cmp::min(PATCH_SIZE, orig_h - y);
            patches.push((x, y, w, h));
            if x + step >= orig_w {
                break;
            }
            x += step;
        }
        if y + step >= orig_h {
            break;
        }
        y += step;
    }
    patches
}

fn preprocess_patch(patch: &DynamicImage, model_size: usize) -> Vec<f32> {
    let resized = patch.resize_exact(
        model_size as u32,
        model_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    let mut data = Vec::with_capacity(3 * model_size * model_size);
    for p in rgb.pixels() {
        data.push((p.0[0] as f32 / 255.0 - mean[0]) / std[0]);
        data.push((p.0[1] as f32 / 255.0 - mean[1]) / std[1]);
        data.push((p.0[2] as f32 / 255.0 - mean[2]) / std[2]);
    }
    data
}

fn postprocess_patch(output: &Array<f32, IxDyn>, orig_w: u32, orig_h: u32) -> Vec<u8> {
    let model_size = output.shape()[2];
    let num_classes = output.shape()[1];
    let mut model_mask = vec![0u8; model_size * model_size];

    for y in 0..model_size {
        for x in 0..model_size {
            let mut max_prob = 0.0;
            let mut class_id = 0;
            for c in 0..num_classes {
                let prob = output[[0, c, y, x]];
                if prob > max_prob {
                    max_prob = prob;
                    class_id = c;
                }
            }
            model_mask[y * model_size + x] = class_id as u8;
        }
    }

    let mask_img = ImageBuffer::from_vec(model_size as u32, model_size as u32, model_mask)
        .unwrap()
        .resize_exact(orig_w, orig_h, image::imageops::FilterType::Nearest);
    mask_img.into_vec()
}

fn merge_patch_mask(
    patch_mask: &[u8],
    x_start: u32,
    y_start: u32,
    patch_w: u32,
    patch_h: u32,
    orig_w: u32,
    global_mask: &mut [u8],
    counts: &mut [u32],
) {
    for (y, row) in patch_mask.chunks(patch_w as usize).enumerate() {
        let global_y = y_start + y as u32;
        if global_y >= orig_w {
            break;
        }
        for (x, &class) in row.iter().enumerate() {
            let global_x = x_start + x as u32;
            if global_x >= orig_w {
                break;
            }
            let idx = (global_y * orig_w + global_x) as usize;
            global_mask[idx] = class;
            counts[idx] += 1;
        }
    }
}

fn save_final_mask(mask: &[u8], w: u32, h: u32, path: &str) -> Result<(), image::ImageError> {
    let mut img = ImageBuffer::new(w, h);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let class = mask[(y * w + x) as usize];
        *p = match class {
            0 => Rgba([0, 0, 0, 255]),
            1 => Rgba([0, 255, 0, 255]),
            2 => Rgba([255, 0, 0, 255]),
            _ => Rgba([255, 255, 0, 255]),
        };
    }
    img.save(path)
}
