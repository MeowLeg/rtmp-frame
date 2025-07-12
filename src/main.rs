use anyhow::{Context, Result};
use std::{
    fs::{read, File},
    io::Read, path::PathBuf,
};
use ffmpeg_next::{
    codec, format, frame, media, software::scaling
};
use std::time::Instant;
use image::{Rgb, ImageBuffer};
use serde::Deserialize;

mod process;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub swim: Option<String>,
    pub fire: Option<String>,
}

pub fn read_from_toml(f: &str) -> Result<Config> {
    let mut file = File::open(f)?;
    let mut s = String::new();
    file.read_to_string(&mut s)?;
    let config: Config = toml::from_str(&s)?;
    Ok(config)
}

fn main() -> Result<()> {
    let cfg = read_from_toml("./config.toml")?;
    // 解析命令行参数
    // let rtmp_url = env::args().nth(1).expect("请提供RTMP URL");
    let rtmp_url = "rtmp://play-sh13.quklive.com/live/1699001836208185?auth_key=2067469022-47d1a627576a4ecf9d5c2068f274f5b0-0-c805cc1b4e4c9e4f51705d2304687f35";

    // 初始化FFmpeg
    ffmpeg_next::init().context("FFmpeg初始化失败")?;

    // 打开RTMP输入流
    let mut ictx = format::input(&rtmp_url)
        .context("无法打开RTMP流")?;

    // 查找视频流
    let video_stream_index = ictx
        .streams()
        .best(media::Type::Video)
        .ok_or_else(|| anyhow::anyhow!("找不到视频流"))?
        .index();

    // 获取视频流
    let stream = ictx
        .streams()
        .find(|s| s.index() == video_stream_index)
        .ok_or_else(|| anyhow::anyhow!("无法获取指定索引的视频流"))?;

    // 创建解码器上下文
    let dctx = codec::context::Context::from_parameters(stream.parameters())
        .context("创建解码器上下文失败")?;
    // 创建缩放上下文，用于将YUV帧转换为RGB

    // 帧计数器和计时器
    let mut frame_count = 0;
    let start_time = Instant::now();

    let mut v_dctx = dctx.decoder().video()?;

    let mut scaler = scaling::Context::get(
        v_dctx.format(),
        v_dctx.width(),
        v_dctx.height(),
        ffmpeg_next::format::Pixel::RGB24,
        v_dctx.width(),
        v_dctx.height(),
        scaling::Flags::BILINEAR,
    )
    .context("无法创建缩放上下文")?;

    // 处理输入包
    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            // 将包发送到解码器
            v_dctx.send_packet(&packet).context("发送数据包失败")?;

            // 从解码器接收帧
            let mut decoded_frame = frame::Video::empty();
            while let Ok(()) = v_dctx.receive_frame(&mut decoded_frame) {
                frame_count += 1;

                // 在这里处理解码后的帧
                let width = decoded_frame.width();
                let height = decoded_frame.height();
                let format = decoded_frame.format();
                
                // 示例：打印帧信息
                if frame_count % 30 == 0 { // 每30帧打印一次
                    let elapsed = start_time.elapsed().as_secs_f64();
                    println!("已处理 {} 帧 | 帧率: {:.2} FPS | 尺寸: {}x{} | 格式: {:?}",
                        frame_count, 
                        frame_count as f64 / elapsed,
                        width, height, format);
                }

                // 自定义帧处理逻辑...
                process_frame(&mut scaler, &decoded_frame, frame_count, &cfg)?;
            }
        }
    }

    // 冲洗解码器
    v_dctx.flush();

    let elapsed = start_time.elapsed().as_secs_f64();
    println!("处理完成: 在 {:.2} 秒内处理了 {} 帧 | 平均帧率: {:.2} FPS",
        elapsed, frame_count, frame_count as f64 / elapsed);

    Ok(())
}

// 自定义帧处理函数（示例）
fn process_frame(
    scaler: &mut scaling::Context,
    frame: &frame::Video,
    frame_count: u32,
    cfg: &Config
) -> Result<Vec<String>> {
    // 这里可以添加自定义处理逻辑
    // 例如：转换格式、分析内容、保存为图片等
    
    // 示例：获取帧的基本信息
    let width = frame.width() as u32;
    let height = frame.height() as u32;
    // let format = frame.format();
    
    // 示例：获取YUV数据
    // let y_data = frame.data(0);  // Y分量
    // let u_data = frame.data(1);  // U分量
    // let v_data = frame.data(2);  // V分量
    
    // 实际应用中，可以在这里添加更复杂的处理逻辑
    // println!("data: width: {} height: {} format: {:?}", width, height, format);
    // println!("data: y_data: {:?}", y_data);
    // println!("data: u_data: {:?}", u_data);
    // println!("data: v_data: {:?}", v_data);
    // 转换帧格式为RGB

    let mut rgb_frame = frame::Video::empty();
    scaler.run(&frame, &mut rgb_frame)
        .context("帧格式转换失败")?;

    let mut image_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);

    // 复制像素数据到图像缓冲区
    let data = rgb_frame.data(0);
    let stride = rgb_frame.stride(0);

    for y in 0..height {
        for x in 0..width {
            let offset = (y as usize * stride) + (x as usize * 3);
            let r = data[offset];
            let g = data[offset + 1];
            let b = data[offset + 2];
            image_buffer.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    
    let filepath = format!("frame_{}.jpg", frame_count);
    image_buffer
        .save(&filepath)
        .context(format!("保存图像失败: {:?}", &filepath))?;

    predicts(cfg, &PathBuf::from(&filepath))
}

fn predicts(cfg: &Config, f: &PathBuf) -> Result<Vec<String>> {
    let mut predicts: Vec<String> = vec![];
    if let Some(m) = &cfg.swim {
        let prediction = process::swim::swim_predict(&m, f)?;
        predicts.push(prediction);
    }

    Ok(predicts)
}