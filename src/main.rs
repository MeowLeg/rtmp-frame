use anyhow::{Context, Result, anyhow};
use axum::{
    Extension,
    routing::{Router, get},
};
use chrono::Local;
use clap::{Parser, Subcommand};
use ffmpeg_next::{codec, format, frame, media, software::scaling};
use image::{ImageBuffer, Rgb};
use ort::session::Session;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use sqlx::{Connection, FromRow, SqliteConnection, sqlite::Sqlite};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    // process::Command,
    sync::Arc,
    time::Instant,
};
use tower_http::services::ServeDir;

mod handler;
mod predict;
mod stream;
use handler::*;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub port: u32,
    pub redis_svr_url: String,
    pub db_path: String,
    pub dump_path: String,
    pub predict_worker_num: u32,
    pub notify_svr_url: String,
    pub notify_timeout: u64,
    pub redis_stream_tag: String,
    pub static_dir: String,
    pub svr_root_url: String,
    pub predict: Vec<Predict>,
    pub is_test: bool,
    pub frame_interval_count: u32,
}

#[derive(Debug, Deserialize)]
pub struct Predict {
    pub tag: String,
    pub title: String,
    pub content: String,
    pub model: String,
    pub pipe: String,
    pub imgsz: usize,
    pub label: Vec<String>,
}

#[derive(Debug, Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// 处理的rtmp流地址
    #[arg(short, long)]
    url: Option<String>,

    /// url md5 value
    #[arg(short, long)]
    md5: Option<String>,

    /// config file
    #[arg(short, long, default_value = "./config.toml")]
    config: String,

    #[arg(short, long, default_value = "")]
    project_uuid: String,

    #[arg(short, long, default_value = "")]
    organization_uuid: String,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// run as daemon for monitor
    Web,

    /// 预测子进程应与Web服务在一起，此处保留单独命令，用于方便增减
    Predict,
}

pub fn read_from_toml(f: &str) -> Result<Config> {
    let mut file = File::open(f)?;
    let mut s = String::new();
    file.read_to_string(&mut s)?;
    let config: Config = toml::from_str(&s)?;
    Ok(config)
}

async fn web_svr(cfg: &Arc<Config>) -> Result<()> {
    let app = Router::new()
        .nest_service("/static", ServeDir::new(&cfg.static_dir))
        .route("/", get(async || "hello, msg data!".to_string()))
        .route("/new_stream", get(new_stream::NewStream::handle_get))
        .layer(Extension(Arc::clone(cfg)));
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", cfg.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[allow(unused)]
pub fn get_current_str(concat: Option<&str>) -> String {
    let now = Local::now();
    let fmt = match concat {
        Some(c) => &format!("%Y{c}%m{c}%d{c}%H{c}%M{c}%S"),
        None => "%Y-%m-%d %H:%M:%S",
    };
    now.format(fmt).to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Arc::new(read_from_toml(&cli.config)?);
    // println!("cli is {:?}", cli);
    // let rds = Client::open(cfg.redis_svr_url.clone())?;

    match cli.command {
        Some(Commands::Web) => {
            // for _ in 0..cfg.predict_worker_num {
            //     let _ = Command::new("./rtmp-frame").arg("predict").spawn()?;
            // }
            // web仅仅用于接收启动与停止命令
            // 预测可以另写一个单独的处理进程
            return web_svr(&cfg).await;
        }
        Some(Commands::Predict) => {
            // 预测应该对数据库执行，减少对redis的依赖
            // 去掉redis
            return predict::predict(&cfg).await;
        }
        None => {}
    }

    if cfg.is_test {
        if let Some(url) = cli.url {
            let stream_md5_val = format!("{:x}", md5::compute(url.as_bytes()));
            return stream::stream(
                Arc::clone(&cfg),
                &url,
                // &rds,
                &stream_md5_val,
                &cli.project_uuid,
                &cli.organization_uuid,
            )
            .await;
        }
    } else {
        if let Some(url) = cli.url
            && let Some(stream_md5_val) = cli.md5
            && stream_md5_val == format!("{:x}", md5::compute(url.as_bytes()))
        {
            return stream::stream(
                Arc::clone(&cfg),
                &url,
                // &rds,
                &stream_md5_val,
                &cli.project_uuid,
                &cli.organization_uuid,
            )
            .await;
        }
    }

    Ok(())
}
