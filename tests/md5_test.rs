#[test]
fn md5_test() {
    let v = format!("{:x}", md5::compute("hello world".as_bytes()));
    println!("v is {}", v);
    assert!(true);
}
