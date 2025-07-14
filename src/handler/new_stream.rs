use super::*;
use std::process::Command;
use sysinfo::System;

pub struct NewStream;

#[derive(Debug, Deserialize)]
pub struct NewStreamReq {
    stream_url: String,
}


impl ExecSql<NewStreamReq> for NewStream {
    async fn handle_get(
        _cfg: Extension<Arc<Config>>,
        prms: Option<Query<NewStreamReq>>
    ) -> Result<Json<Value>, WebErr> {
        let prms= match prms {
            Some(Query(p)) => p,
            None => {
                return Err("no stream url".into());
            }
        };

        let mut success = false;
        let md5_val = format!("{:x}", md5::compute(prms.stream_url.as_bytes()));
        if !is_stream_valid(&md5_val)? {
            success = false;
            let _ = Command::new("./rtmp-frame")
                .args(["--url", &prms.stream_url])
                .args(["--md5", &md5_val])
                .spawn()?;
        }
        
        Ok(Json(json!({
            "success": success,
            "errMsg": "",
            "data": &md5_val
        })))
    }
}

fn is_stream_valid(md5_val: &str) -> Result<bool> {
    let sys = System::new_all();
    for (_, p) in sys.processes() {
        if let Some(s) = p.name().to_str() {
            if s.contains("rtmp-frame") {
                if p.cmd().iter().any(|s| s.to_string_lossy().contains(md5_val)) {
                    return Ok(true)
                }
            }
        };
    }
    Ok(false)
}