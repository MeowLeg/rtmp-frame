use super::*;

pub struct StopDump;

#[derive(Debug, Deserialize, Serialize)]
pub struct StopDumpReq {
    uuid: String,
}

impl ExecSql<StopDumpReq> for StopDump {
    async fn handle_post(
        cfg: Extension<Arc<Config>>,
        params: Result<Json<StopDumpReq>, JsonRejection>,
    ) -> Result<Json<Value>, WebErr> {
        let Json(prms) = params?;
        let mut conn = SqliteConnection::connect(&cfg.db_path).await?;

        let sys = System::new_all();
        for p in sys.processes().values() {
            if let Some(s) = p.name().to_str()
                && s.contains(&prms.uuid)
            {
                let b = p.kill();
                let _ = sqlx::query("update stream set is_over = 1 where uuid = ?")
                    .bind(&prms.uuid)
                    .execute(&mut conn)
                    .await?;
                return Ok(Json(json!({
                    "success": b,
                })));
            }
        }

        Ok(Json(json!({
            "success": false,
        })))
    }
}
