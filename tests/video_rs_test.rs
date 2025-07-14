use anyhow::{
    Result,
    anyhow,
};

#[test]
fn test_video_rs() -> Result<()> {
    use video_rs::decode::Decoder;
    use video_rs::Url;

    video_rs::init().map_err(|e| anyhow!("failed to init video_rs: {:?}", e))?;

    let rtmp_url = "rtmp://play-sh13.quklive.com/live/1699001836208185?auth_key=2067469022-47d1a627576a4ecf9d5c2068f274f5b0-0-c805cc1b4e4c9e4f51705d2304687f35";
    let source = rtmp_url 
        .parse::<Url>()?;
    let mut decoder = Decoder::new(source)?;

    for frame in decoder.decode_iter() {
        if let Ok((_, frame)) = frame {
            let rgb = frame.slice(ndarray::s![0, 0, ..]).to_slice().unwrap();
            println!("pixel at 0, 0: {}, {}, {}", rgb[0], rgb[1], rgb[2],);
        } else {
            break;
        }
    }
    Ok(())
}