
use super::*;

const SWIM_LABELS: [&str; 1] = [
    "person",
];

pub fn swim_predict(m: &mut Session, im: &PathBuf) -> Result<String> {
    Ok(predict(m, im, "swim", &SWIM_LABELS)?)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() -> Result<()> {
        let model_path = "./model/swim_yolo8_nano.onnx";
        let mut model = match Session::builder()?
        .commit_from_file(model_path) {
            Ok(m) => m,
            Err(e) => {
                println!("Error loading model: {}", e);
                return Err(e.into());
            }
        };
        for im in std::fs::read_dir("./data/swim").unwrap() {
            let im = im.unwrap();
            if im.file_type().unwrap().is_file() {
                let start = get_current_str();
                swim_predict(&mut model, &im.path()).unwrap();
                let end = get_current_str();
                println!("Start: {}, End: {}", start, end);
            }
        }
        assert!(true);
        Ok(())
    }
}
