use super::*;
use tokio::time::{Duration, sleep};

// pub async fn watch(cmd: &str, interval: u64, max_timeout: u64) -> Result<()> {
pub async fn watch(cfg: Arc<Config>) -> Result<()> {
    loop {
        let sys = System::new_all();
        for p in sys.processes().values() {
            if let Some(s) = p.name().to_str()
                && s.contains(&cfg.main_cmd)
            {
                if p.run_time() > cfg.rtmp_max_timeout {
                    let _ = p.kill();
                }
            }
        }

        sleep(Duration::from_secs(cfg.watch_interval)).await;
    }
}
