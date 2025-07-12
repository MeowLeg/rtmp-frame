
use super::*;

const YOLOV8_CLASS_LABELS: [&str; 1] = [
    "person",
];

pub fn swim_predict(m: &str, im: &PathBuf) -> Result<String> {
    Ok(predict(m, im, "swim", YOLOV8_CLASS_LABELS)?)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        for im in std::fs::read_dir("./data/swim").unwrap() {
            let im = im.unwrap();
            if im.file_type().unwrap().is_file() {
                let start = get_current_str();
                predict("./model/swim_yolo8_nano.onnx", &im.path(), "swim").unwrap();
                let end = get_current_str();
                println!("Start: {}, End: {}", start, end);
            }
        }
        assert!(true);
    }
}