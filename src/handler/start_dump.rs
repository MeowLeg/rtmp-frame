use super::*;
use std::process::Command;

pub struct StartDump;

#[derive(Debug, Deserialize, Serialize)]
pub struct StartDumpReq {
    uuid: String,
    organization_uuid: String,
    project_uuid: String,
    flight_uuid: String,
    sn: String,
    rtmp: String,
    labels: Vec<Label>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Label {
    id: u32,
    code: String,
    name: String,
    tenant_id: u32,
    created_at: Option<u128>,
    updated_at: Option<u128>,
    deleted_at: Option<u128>,
}

impl ExecSql<StartDumpReq> for StartDump {
    async fn handle_post(
        Extension(cfg): Extension<Arc<Config>>,
        params: Result<Json<StartDumpReq>, JsonRejection>,
    ) -> Result<Json<Value>, WebErr> {
        let Json(prms) = params?;
        let _ = start_rtmp(
            &prms.rtmp,
            &prms.uuid,
            &prms.project_uuid,
            &prms.organization_uuid,
        )?;
        let _ = log_stream(&cfg.db_path, &prms).await?;
        Ok(Json(json!({
            "success": true
        })))
    }
}

fn start_rtmp(url: &str, uuid: &str, project_uuid: &str, organization_uuid: &str) -> Result<()> {
    let _ = Command::new("./rtmp-frame")
        .args(["--url", url])
        .args(["--uuid", uuid])
        .args(["--project_uuid", project_uuid])
        .args(["--organization_uuid", organization_uuid])
        .spawn()?;

    Ok(())
}

async fn log_stream(db_path: &str, req: &StartDumpReq) -> Result<i64> {
    let mut conn = SqliteConnection::connect(db_path).await?;
    let sql = r#"
        insert into stream(uuid, stream_url, project_uuid, organization_uuid)
        values(?,?,?)
        "#;
    let r = sqlx::query(sql)
        .bind(&req.uuid)
        .bind(&req.rtmp)
        .bind(&req.project_uuid)
        .bind(&req.organization_uuid)
        .execute(&mut conn)
        .await?;

    let tag_sql = r#"
        insert into stream_tag(uuid, code, name)
        "#;
    for l in req.labels.iter() {
        let _ = sqlx::query(tag_sql)
            .bind(&req.uuid)
            .bind(&l.code)
            .bind(&l.name)
            .execute(&mut conn)
            .await?;
    }

    Ok(r.last_insert_rowid())
}
