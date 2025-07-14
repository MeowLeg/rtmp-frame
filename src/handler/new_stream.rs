use super::*;
use std::process::Command;
use sysinfo::System;

pub struct NewStream;

#[derive(Debug, Deserialize, Serialize)]
pub struct NewStreamReq {
    pub stream_url: String,
    pub project_uuid: Option<String>,
    pub organization_uuid: Option<String>,
}


impl ExecSql<NewStreamReq> for NewStream {
    async fn handle_get_with_redis(
        cfg: Extension<Arc<Config>>,
        rds: Extension<Arc<Client>>,
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

        let mut con = rds.get_multiplexed_async_connection().await?;
        con.set::<_, String, String>(
            format!("{}_{}", &cfg.redis_stream_tag, md5_val),
            serde_json::to_string(&prms)?
        ).await?;

        Ok(Json(json!({
            "success": success,
            "errMsg": "",
            "data": &md5_val
        })))
    }
}

#[allow(unused)]
fn is_stream_valid(md5_val: &str) -> Result<bool> {
    let sys = System::new_all();
    for p in sys.processes().values() {
        if let Some(s) = p.name().to_str()
            && s.contains("rtmp-frame")
            && p.cmd().iter().any(|s| s.to_string_lossy().contains(md5_val)) {
                return Ok(true)
            }
    }
    Ok(false)
}