use anyhow::{
    Context, Result,
    anyhow,
};
use std::{
    fs::File,
    io::Read,
    path::PathBuf,
};
use ffmpeg_next::{
    codec, format, frame, media, software::scaling
};
use std::time::Instant;
use image::{Rgb, ImageBuffer};
use serde::Deserialize;
use ort::session::Session;
use axum::{
    Extension,
    routing::{
        Router,
        get,
        post,
    }
};
use std::sync::{
    Arc,
    Mutex,
};
use clap::{
    Parser, Subcommand
};
use chrono::Local;
use redis::{
    AsyncCommands,
    Client,
    Pipeline,
};

mod predict;
mod stream;
mod handler;
use handler::*;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub port: u32,
    pub redis_url: String,
    pub dump_path: String,
    pub predict: Vec<Predict>,
}

#[derive(Debug, Deserialize)]
pub struct Predict {
    pub tag: String,
    pub model: String,
    pub pipe: String,
    pub label: Vec<String>,
}

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// run edge for incrment of duration or out pv
    #[arg(short, long)]
    url: Option<String>,

    /// url md5 value
    #[arg(short, long)]
    md5: Option<String>,

    /// config file
    #[arg(short, long, default_value="./config.toml")]
    config: String,
}

#[derive(Subcommand)]
enum Commands {
    /// run as daemon for monitor
    Web,

    // predict img from redis pipe
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
        .route("/", get(async || "hello, msg data!".to_string()))
        .route("/new_stream", get(new_stream::NewStream::handle_get))
        .layer(Extension(Arc::clone(&cfg)));
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", cfg.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[allow(unused)]
pub fn get_current_str(concat: Option<&str>) -> String {
    let now = Local::now();
    let fmt = match concat {
        Some(c) => &format!("%Y{}%m{}%d{}%H{}%M{}%S", c, c, c, c, c),
        None => "%Y-%m-%d %H:%M:%S",
    };
    let today = now.format(fmt).to_string();
    today
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Arc::new(read_from_toml("./config.toml")?);
    let cli = Cli::parse();
    let rds = Client::open(cfg.redis_url.clone())?;

    match cli.command {
        Some(Commands::Web) => {
            return web_svr(&cfg).await;
        },
        Some(Commands::Predict) => {
            return predict::predict(&cfg, &rds).await;
            todo!("应该在这里启动一系列后台运行的服务，因此不应该已子命令的形式出现");
        },
        None => {}
    }

    if let Some(url) = cli.url {
        if let Some(md5_val) = cli.md5 {
            if md5_val == format!("{:x}", md5::compute(url.as_bytes())) {
                return stream::stream(&url, Arc::clone(&cfg), &rds).await;
            }
        }
    }

    Ok(())
}