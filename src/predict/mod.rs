use super::*;

use image::{
    GenericImageView,
};
use ndarray::{
	Array,
	Axis,
	s,
};
use ort::{
	inputs,
	value::TensorRef
};
use ab_glyph::{
	FontRef,
	PxScale,
};
use imageproc::drawing::{
	draw_text_mut,
	draw_hollow_rect_mut,
};
use std::path::PathBuf;
use chrono::Local;


#[allow(unused)]
use ort::execution_providers::*;

#[allow(unused)]
pub fn init() -> ort::Result<()> {
	#[cfg(feature = "backend-candle")]
	ort::set_api(ort_candle::api());
	#[cfg(feature = "backend-tract")]
	ort::set_api(ort_tract::api());

	#[cfg(all(not(feature = "backend-candle"), not(feature = "backend-tract")))]
	ort::init()
		.with_execution_providers([
			#[cfg(feature = "tensorrt")]
			TensorRTExecutionProvider::default().build(),
			#[cfg(feature = "cuda")]
			CUDAExecutionProvider::default().build(),
			#[cfg(feature = "onednn")]
			OneDNNExecutionProvider::default().build(),
			#[cfg(feature = "acl")]
			ACLExecutionProvider::default().build(),
			#[cfg(feature = "openvino")]
			OpenVINOExecutionProvider::default().build(),
			#[cfg(feature = "coreml")]
			CoreMLExecutionProvider::default().build(),
			#[cfg(feature = "rocm")]
			ROCmExecutionProvider::default().build(),
			#[cfg(feature = "cann")]
			CANNExecutionProvider::default().build(),
			#[cfg(feature = "directml")]
			DirectMLExecutionProvider::default().build(),
			#[cfg(feature = "tvm")]
			TVMExecutionProvider::default().build(),
			#[cfg(feature = "nnapi")]
			NNAPIExecutionProvider::default().build(),
			#[cfg(feature = "qnn")]
			QNNExecutionProvider::default().build(),
			#[cfg(feature = "xnnpack")]
			XNNPACKExecutionProvider::default().build(),
			#[cfg(feature = "armnn")]
			ArmNNExecutionProvider::default().build(),
			#[cfg(feature = "migraphx")]
			MIGraphXExecutionProvider::default().build(),
			#[cfg(feature = "vitis")]
			VitisAIExecutionProvider::default().build(),
			#[cfg(feature = "rknpu")]
			RKNPUExecutionProvider::default().build(),
			#[cfg(feature = "webgpu")]
			WebGPUExecutionProvider::default().build()
		])
		.commit()?;

	Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
	x1: f32,
	y1: f32,
	x2: f32,
	y2: f32
}

#[allow(unused)]
fn _predict(model: &mut Session, im_path: &PathBuf, tag: &str, labels: &[String]) -> Result<String> {
    init()?;

    let nm = format!("{}_{}",
		im_path 
			.file_stem()
			.ok_or(anyhow::anyhow!("Failed to get file stem"))?
			.to_string_lossy()
			.to_string(),
		tag
	);

    // let mut model= match Session::builder()?
    //     .commit_from_file(model_path) {
    //         Ok(m) => m,
    //         Err(e) => {
    //             println!("Error loading model: {}", e);
    //             return Err(e.into());
    //         }
    //     };

    let img = match image::open(&im_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error opening image: {}", e);
            return Err(e.into());
        }
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    // println!("w is {}, h is {}", w, h);


    let re_img = img.resize_exact(640, 640, image::imageops::FilterType::CatmullRom);

    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in re_img.pixels() {
        let x = pixel.0 as _;
		let y = pixel.1 as _;
		let [r, g, b, _] = pixel.2.0;
		input[[0, 0, y, x]] = (r as f32) / 255.;
		input[[0, 1, y, x]] = (g as f32) / 255.;
		input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let input_values = match TensorRef::from_array_view(&input) {
        Ok(values) => values,
        Err(e) => {
            println!("Error converting input to TensorRef: {}", e);
            return Err(e.into());
        }
    };
    let outputs = model.run(
        inputs!["images" => input_values]
    )?;

    let output = outputs["output0"].try_extract_array::<f32>()?.t().into_owned();
    // println!("{:?}", output);

    let mut boxes = Vec::new();
	let output = output.slice(s![.., .., 0]);
	for row in output.axis_iter(Axis(0)) {
		let row: Vec<_> = row.iter().copied().collect();
		let (class_id, prob) = row
			.iter()
			// skip bounding box coordinates
			.skip(4)
			.enumerate()
			.map(|(index, value)| (index, *value))
			.reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
			.unwrap();
		if prob < 0.5 {
			continue;
		}
		let label = labels[class_id].as_ref();
		let xc = row[0] / 640. * (w as f32);
		let yc = row[1] / 640. * (h as f32);
		let w = row[2] / 640. * (w as f32);
		let h = row[3] / 640. * (h as f32);
        let b = BoundingBox {
            x1: xc - w / 2.,
            y1: yc - h / 2.,
            x2: xc + w / 2.,
            y2: yc + h / 2.
        };
		boxes.push((
            b,
			label,
			prob
		));
        println!("class_id: {}, prob: {}, label: {}, boxes: {:?}", class_id, prob, label, boxes);
	}
    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));

    println!("Detected {} objects:", boxes.len());
    if !boxes.is_empty() || boxes.len() > 0 {
        let _ = visualize_detections(&im_path, boxes, &nm)?;
    }

    Ok(nm.to_string())
}

#[allow(unused)]
fn visualize_detections(image_path: &PathBuf, boxes: Vec<(BoundingBox, &str, f32)>, nm: &str) -> Result<()> {
    // Load the original image
    let mut img = image::open(image_path)?.to_rgb8();

    let font = FontRef::try_from_slice(include_bytes!("../../font/Deng.ttf"))?;
    let scale = PxScale { x: 20.0 as f32, y: 20.0 as f32 };

    for (bbox, label, prob) in boxes {
        // Draw bounding box
        let rect = imageproc::rect::Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);
        draw_hollow_rect_mut(
            &mut img,
            rect, 
            Rgb([255, 0, 0])
        );

        // Draw label and probability
        let text = format!("{} ({:.2})", label, prob);
        draw_text_mut(
            &mut img, 
            Rgb([255, 255, 255]), 
            bbox.x1 as i32, 
            bbox.y1 as i32 - 20,
            scale,
            &font,
            &text
        );
    }

    // Save the image with visualized detections
    let output_path = image_path.with_file_name(format!("{}_out.jpg", nm));
    img.save(&output_path)?;

    println!("Visualized detections saved to {:?}", output_path);
    Ok(())
}

pub async fn predict(cfg: &Config, rds: &Client) -> Result<()> {
    let mut con = rds.get_multiplexed_async_connection().await?;
    loop {
        for p in cfg.predict.iter() {
            let f: String = con.rpop(&p.pipe, None).await?;
            let mut m = Session::builder()?
                .with_execution_providers([CUDAExecutionProvider::default().build()])?
                .commit_from_file(&p.model)?;
            _predict(&mut m, &PathBuf::from(f), &p.tag, &p.label)?;
            todo!("需要图片名发送给某个服务");
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ort::execution_providers::cuda::CUDAExecutionProvider;

    fn swim_predict(m: &mut Session, im: &PathBuf) -> Result<String> {
        Ok(_predict(m, im, "swim", &["person".into()])?)
    }

    #[test]
    fn test_predict() -> Result<()> {
        let model_path = "./model/swim_yolo8_nano.onnx";
        let mut model = match Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(model_path) {
            Ok(m) => m,
            Err(e) => {
                println!("Error loading model: {}", e);
                return Err(e.into());
            }
        };
        for im in std::fs::read_dir("./data/swim")? {
            let im = im?;
            if im.file_type()?.is_file() {
                let start = get_current_str(None);
                swim_predict(&mut model, &im.path())?;
                let end = get_current_str(None);
                println!("Start: {}, End: {}", start, end);
            }
        }
        assert!(true);
        Ok(())
    }
}