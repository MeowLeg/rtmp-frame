use super::*;
use ab_glyph::{FontRef, PxScale};
use image::GenericImageView;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use ndarray::{Array, Axis, s};
use ort::execution_providers::*;
use ort::{inputs, value::TensorRef};
use serde_json::{Value, json};
use std::{fs::create_dir_all, path::PathBuf};
pub mod div_predict;

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
            WebGPUExecutionProvider::default().build(),
        ])
        .commit()?;

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[allow(unused)]
fn _predict(
    model: &mut Session,
    im_path: &PathBuf,
    imgsz: usize,
    tag: &str, // 模型的标签，目前没有什么用处
    labels: &[String],
    out_pf: &PathBuf,
) -> Result<()> {
    // println!("start model-------------------0");
    // init()?;

    let img = match image::open(im_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error opening image: {e}");
            return Err(e.into());
        }
    };
    // println!("start model-------------------1");
    let (w, h) = (img.width() as usize, img.height() as usize);
    // println!("w is {}, h is {}", w, h);

    // let re_img = img.resize_exact(imgsz as u32, imgsz as u32, image::imageops::FilterType::CatmullRom);
    let re_img = img.resize(
        imgsz as u32,
        imgsz as u32,
        image::imageops::FilterType::Nearest,
    );
    // println!("start model-------------------1.5");

    let mut input = Array::zeros((1, 3, imgsz, imgsz));
    // println!("input before is {:?}", &input);
    // let mut i = 0;
    for pixel in re_img.pixels() {
        // if i < 100 {
        //     println!("pixel is {:?}", &pixel);
        //     i += 1;
        // }
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    // println!("0, 1, 300, 300 is {:?}", &input[[0, 1, 300, 300]]);
    // println!("input after is {:?}", &input);
    // println!("start model-------------------2");

    let input_values = match TensorRef::from_array_view(&input) {
        Ok(values) => values,
        Err(e) => {
            println!("Error converting input to TensorRef: {e}");
            return Err(e.into());
        }
    };
    // println!("tensor array is {:?}", &input_values);
    // println!("start model-------------------3");
    // let outputs = model.run(inputs!["images" => input_values])?;
    let outputs = model.run(inputs![input_values])?;
    // println!("start model-------------------4");
    // println!("outputs is {:?}", &outputs);

    // println!("outputs: {:?}", outputs); // when test

    let output = outputs["output0"]
        .try_extract_array::<f32>()?
        .t()
        .into_owned();
    // println!("start model-------------------5");
    // println!("output is {:?}", &output);

    // println!("output: {:?}", &output); // when test

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

        println!("output is {:?}", &output); // test
        println!("row is {:?}", &row); // test

        let imgsz_ft = imgsz as f32;
        let label = labels[class_id].as_ref();
        // println!("label is {}", &label); // test
        let xc = row[0] / imgsz_ft * (w as f32);
        let yc = row[1] / imgsz_ft * (h as f32);
        let w = row[2] / imgsz_ft * (w as f32);
        let h = row[3] / imgsz_ft * (h as f32);
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

    println!("Detected {} objects:", boxes.len());
    if !boxes.is_empty() {
        visualize_detections(im_path, boxes, out_pf);
        return Ok(());
    }

    Err(anyhow!("no detect"))
}

fn visualize_detections(
    image_path: &PathBuf,
    boxes: Vec<(BoundingBox, &str, f32)>,
    out_pf: &PathBuf,
) -> Result<()> {
    // Load the original image
    let mut img = image::open(image_path)?.to_rgb8();

    let font = FontRef::try_from_slice(include_bytes!("../../font/Deng.ttf"))?;
    let scale = PxScale {
        x: 20.0_f32,
        y: 20.0_f32,
    };

    for (bbox, label, prob) in boxes {
        // Draw bounding box
        let rect = imageproc::rect::Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);
        draw_hollow_rect_mut(&mut img, rect, Rgb([255, 0, 0]));

        // Draw label and probability
        let text = format!("{label} ({prob:.2})");
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

#[derive(Debug, Serialize, FromRow)]
pub struct Pic {
    id: i32,
    path: String,
    stream_url: String,
    project_uuid: String,
    organization_uuid: String,
    pic_md5: String,
    create_date: String,
    predicted: i32,
}

#[derive(Debug, Serialize)]
struct NotifyData {
    alarm_uuid: String,
    title: String,
    content: String,
    alarm_type: String,
    #[serde(rename = "dateTime")]
    date_time: String,
    #[serde(rename = "mediaUrl")]
    media_url: Vec<String>,
    #[serde(rename = "videoUrl")]
    video_url: Vec<String>,
    organization_uuid: String,
    project_uuid: String,
    rtmp: String,
}

fn get_uuid() -> String {
    let u = uuid::Uuid::new_v4();
    format!("{u}").split("-").collect::<String>()
}

// pub async fn predict(cfg: &Config, rds: &Client) -> Result<()> {
pub async fn predict(cfg: &Config) -> Result<()> {
    // let mut con = rds.get_multiplexed_async_connection().await?;
    init()?;
    let mut conn = SqliteConnection::connect(&cfg.db_path).await?;
    let sql = r#"
            select id, path, stream_url,
                project_uuid, organization_uuid, pic_md5,
                create_date, predicted
            from pic
            where predicted = 0
        "#;
    let pics = sqlx::query_as::<Sqlite, Pic>(sql)
        .fetch_all(&mut conn)
        .await?;

    for p in cfg.predict.iter() {
        // let f: String = con.rpop(&p.pipe, None).await?;
        for pic in pics.iter() {
            // println!("pic is {:?}", &pic); test
            // let md5_val = f.split("_").next().ok_or(anyhow!("can not get md5 val"))?;
            // println!("md5 is {}", &md5_val);
            // let data: String = con
            //     .get(format!("{}_{}", &cfg.redis_stream_tag, md5_val))
            //     .await?;
            // println!("data is {}", &data);
            // let stream_info: NewStreamReq = serde_json::from_str(&data)?;
            let mut m = Session::builder()?
                .with_execution_providers([CUDAExecutionProvider::default().build()])?
                .commit_from_file(&p.model)?;

            let pf = PathBuf::from(&pic.path);
            let mut out_pf = PathBuf::from(&cfg.static_dir);
            out_pf = out_pf.join(&p.tag);
            out_pf = out_pf.with_file_name(pf.file_name().ok_or(anyhow!("can not get file name"))?);
            if let Some(p) = out_pf.parent() {
                create_dir_all(p)?;
            }

            match _predict(&mut m, &pf, p.imgsz, &p.tag, &p.label, &out_pf) {
                Ok(()) => {
                    let output_url = format!(
                        "{}/static/{}",
                        cfg.svr_root_url,
                        out_pf.file_name().unwrap().to_string_lossy()
                    );
                    let _ = _notify(
                        &cfg.notify_svr_url,
                        json!(NotifyData {
                            alarm_uuid: get_uuid(),
                            title: p.title.clone(),
                            content: p.content.clone(),
                            alarm_type: p.tag.clone(),
                            date_time: get_current_str(None),
                            media_url: vec![output_url], // todo
                            video_url: vec![],
                            organization_uuid: pic.organization_uuid.clone(),
                            project_uuid: pic.project_uuid.clone(),
                            rtmp: pic.stream_url.clone()
                        }),
                        cfg.notify_timeout,
                    )
                    .await;
                }
                Err(e) => {
                    println!("err is {:?}", e);
                }
            }
            // todo!("需要将本地文件名转为http访问");
        }
    }

    // 更新数据库pic表，表示已经被AI判断过了
    let update_sql = "update pic set predicted = 1 where id = ?";
    for pic in pics.iter() {
        let _ = sqlx::query(update_sql)
            .bind(pic.id)
            .execute(&mut conn)
            .await?;
    }

    Ok(())
}

#[allow(unused)]
async fn _notify(sms_server_url: &str, payload: Value, timeout: u64) -> Result<()> {
    let cli = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout))
        .build()?;
    let _resp = cli.post(sms_server_url).json(&payload).send().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::execution_providers::cuda::CUDAExecutionProvider;

    fn swim_predict(m: &mut Session, im: &PathBuf) -> Result<()> {
        let stem = im.file_stem().unwrap().to_string_lossy();
        let ext = im.extension().unwrap().to_string_lossy();
        let out = format!("{stem}_out.{ext}");
        _predict(
            m,
            im,
            640,
            "swim",
            &[
                "pedestrian".into(),
                "people".into(),
                "bicycle".into(),
                "car".into(),
                "van".into(),
                "truck".into(),
                "tricycle".into(),
                "awning-tricycle".into(),
                "bus".into(),
                "motor".into(),
            ],
            &im.with_file_name(out),
        )
    }

    #[test]
    fn test_predict() -> Result<()> {
        // let model_path = "./model/swim_yolo8_nano.onnx";
        let model_path = "./model/yolo11n_visdrone.onnx";
        let mut model = match Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .commit_from_file(model_path)
        {
            Ok(m) => m,
            Err(e) => {
                println!("Error loading model: {}", e);
                return Err(e.into());
            }
        };

        // for im in std::fs::read_dir("./dump")? {
        //     let im = im?;
        //     if im.file_type()?.is_file() {
        //         let start = get_current_str(None);
        //         let _ = swim_predict(&mut model, &im.path());
        //         let end = get_current_str(None);
        //         println!("Start: {}, End: {}", start, end);
        //     }
        // }

        let im = PathBuf::from("./dump/f353e1b849e5f8f5e6b740359f0c5858_20251029174126_4440.jpg");
        let _ = swim_predict(&mut model, &im);
        assert!(true);
        Ok(())
    }
}
