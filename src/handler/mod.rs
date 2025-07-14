use super::*;
pub mod new_stream;

use std::error::Error;
use axum::{
    Json,
    extract::{
        Query, Extension,
        rejection::JsonRejection,
    },
    response::{
        IntoResponse,
        Response
    },
    http::StatusCode,
};
use serde_json::{
    Value,
    json
};
use http::HeaderMap;

#[allow(dead_code)]
pub trait ExecSql<T> {
    async fn handle_post(
        _cfg: Extension<Arc<Config>>,
        _prms: Result<Json<T>, JsonRejection>
    ) -> Result<Json<Value>, WebErr> {
        Ok(Json(json!({})))
    }

    async fn handle_post_with_redis_cli(
        _cfg: Extension<Arc<Config>>,
        _redis: Extension<Arc<Client>>,
        _prms: Result<Json<T>, JsonRejection>
    ) -> Result<Json<Value>, WebErr> {
        Ok(Json(json!({})))
    }

    async fn handle_get(
        _cfg: Extension<Arc<Config>>,
        _prms: Option<Query<T>>
    ) -> Result<Json<Value>, WebErr> {
        Ok(Json(json!({})))
    }

    async fn handle_get_with_redis(
        _cfg: Extension<Arc<Config>>,
        _redis: Extension<Arc<Client>>,
        _prms: Option<Query<T>>
    ) -> Result<Json<Value>, WebErr> {
        Ok(Json(json!({})))
    }

    async fn handle_get_with_headers(
        _headers: HeaderMap,
        _cfg: Extension<Arc<Config>>,
        _prms: Option<Query<T>>
    ) -> Result<Json<Value>, WebErr> {
        Ok(Json(json!({})))
    }
}

#[derive(Debug)]
pub struct WebErr(Box<dyn Error+Send+Sync>);

impl IntoResponse for WebErr {
    fn into_response(self) -> Response {
        let j = json!({
            "success": false,
            "errMsg": format!("{}", self.0),
            "data": "",
        }).to_string();
        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(j.into()).unwrap()
    }
}

impl<E> From<E> for WebErr
    where E: Into<Box<dyn Error+Send+Sync>>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}