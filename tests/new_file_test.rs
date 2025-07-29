use anyhow::Result;
use md5;
use std::path::PathBuf;

#[test]
fn new_file_test() -> Result<()> {
    let ori_file_path = "./data/swim/334.jpg";
    let ext = ori_file_path.split(".").last().unwrap_or("jpg");
    let md5_val = format!("{:x}", md5::compute(ori_file_path.as_bytes()));
    let img = image::open(ori_file_path)?.to_rgb8();
    let mut pf = PathBuf::from("./static");
    pf = pf.join("xnilan_0001");
    pf = pf.join(format!("{md5_val}.{ext}"));
    if let Some(p) = pf.parent() {
        std::fs::create_dir_all(p)?;
    }
    img.save(pf)?;

    assert!(true);
    Ok(())
}

#[test]
fn rename_test() -> Result<()> {
    let mut pf = PathBuf::from("./static/hello.jpg");
    pf = pf.with_file_name("world.jpg");
    println!("path is {:?}", &pf);
    assert!(true);
    Ok(())
}
