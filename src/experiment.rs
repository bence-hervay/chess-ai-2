//! Experiment run storage: self-contained run directories, environment
//! manifests, and process resource probes.
//!
//! Every run lives in `runs/<timestamp>-<label>-<git-sha>/` and contains the
//! fully resolved configuration, a machine/environment manifest, metrics,
//! game records, and a summary. Nothing about a run may depend on implicit
//! environment state.

use serde::Serialize;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Environment and provenance manifest recorded for every run.
#[derive(Clone, Debug, Serialize)]
pub struct Manifest {
    pub git_commit: String,
    pub git_dirty: bool,
    pub rust_toolchain: String,
    pub cargo_lock_hash: String,
    pub os: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub allocated_threads: usize,
    pub ram_bytes: u64,
    pub build_profile: String,
    pub build_rustflags: String,
    pub model_parameter_count: u64,
    pub seed: u64,
    pub command: Vec<String>,
    pub start_unix_seconds: u64,
    pub end_unix_seconds: Option<u64>,
    pub exit_status: String,
}

/// 64-bit FNV-1a; used to fingerprint `Cargo.lock`, not for security.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn git_output(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn read_proc_field(path: &str, key: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    text.lines()
        .find(|l| l.starts_with(key))
        .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
}

pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `YYYYmmdd-HHMMSS` in UTC, for run-directory names.
pub fn utc_timestamp_compact(unix_secs: u64) -> String {
    let days = unix_secs / 86_400;
    let secs = unix_secs % 86_400;
    // civil-from-days (Howard Hinnant's algorithm), valid for the Unix era.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        y,
        m,
        d,
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// Process CPU time (user + system, all threads) in seconds, from
/// `/proc/self/stat`. Values are in USER_HZ ticks, which the Linux ABI
/// fixes at 100 (see proc(5)).
pub fn process_cpu_seconds() -> Option<f64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    // Fields 14 and 15 (1-based) are utime and stime. They come after the
    // parenthesised comm field, which may itself contain spaces, so split
    // at the last ')'.
    let (_, after_comm) = stat.rsplit_once(')')?;
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) as f64 / 100.0)
}

/// Peak resident set size in bytes, from `/proc/self/status` `VmHWM`.
pub fn peak_rss_bytes() -> Option<u64> {
    let field = read_proc_field("/proc/self/status", "VmHWM")?;
    let kb: u64 = field.split_whitespace().next()?.parse().ok()?;
    Some(kb * 1024)
}

/// Gather the environment manifest at run start.
pub fn collect_manifest(seed: u64, allocated_threads: usize) -> Manifest {
    let cargo_lock_hash = fs::read("Cargo.lock")
        .map(|bytes| format!("fnv1a64:{:016x}", fnv1a64(&bytes)))
        .unwrap_or_else(|_| "unavailable".to_string());
    let os = fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|t| {
            t.lines().find(|l| l.starts_with("PRETTY_NAME=")).map(|l| {
                l.trim_start_matches("PRETTY_NAME=")
                    .trim_matches('"')
                    .to_string()
            })
        })
        .unwrap_or_else(|| std::env::consts::OS.to_string());
    Manifest {
        git_commit: git_output(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
        git_dirty: git_output(&["status", "--porcelain"])
            .map(|s| !s.is_empty())
            .unwrap_or(true),
        rust_toolchain: env!("BUILD_RUSTC_VERSION").to_string(),
        cargo_lock_hash,
        os,
        cpu_model: read_proc_field("/proc/cpuinfo", "model name")
            .unwrap_or_else(|| "unknown".into()),
        logical_cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        allocated_threads,
        ram_bytes: read_proc_field("/proc/meminfo", "MemTotal")
            .and_then(|f| {
                f.split_whitespace()
                    .next()
                    .and_then(|kb| kb.parse::<u64>().ok())
            })
            .map(|kb| kb * 1024)
            .unwrap_or(0),
        build_profile: env!("BUILD_PROFILE").to_string(),
        build_rustflags: env!("BUILD_RUSTFLAGS").to_string(),
        model_parameter_count: 0,
        seed,
        command: std::env::args().collect(),
        start_unix_seconds: unix_seconds(),
        end_unix_seconds: None,
        exit_status: "running".to_string(),
    }
}

/// A self-contained run directory.
pub struct RunDir {
    path: PathBuf,
}

impl RunDir {
    /// Create `runs/<timestamp>-<label>-<sha7>/` (plus `games/`).
    pub fn create(root: &Path, label: &str, git_commit: &str) -> std::io::Result<RunDir> {
        let sha7: String = git_commit.chars().take(7).collect();
        let name = format!(
            "{}-{}-{}",
            utc_timestamp_compact(unix_seconds()),
            label,
            sha7
        );
        let path = root.join(name);
        fs::create_dir_all(path.join("games"))?;
        Ok(RunDir { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write_json<T: Serialize>(&self, name: &str, value: &T) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(value).expect("serializable value");
        fs::write(self.path.join(name), text)
    }

    pub fn write_text(&self, name: &str, text: &str) -> std::io::Result<()> {
        fs::write(self.path.join(name), text)
    }

    pub fn append_jsonl<T: Serialize>(&self, name: &str, values: &[T]) -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path.join(name))?;
        let mut buffer = String::new();
        for value in values {
            buffer.push_str(&serde_json::to_string(value).expect("serializable value"));
            buffer.push('\n');
        }
        file.write_all(buffer.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_formatting_is_correct() {
        // 2026-08-14 00:00:00 UTC == 1786924800? Verify a known epoch:
        // 2020-01-01T00:00:00Z == 1577836800.
        assert_eq!(utc_timestamp_compact(1_577_836_800), "20200101-000000");
        assert_eq!(utc_timestamp_compact(0), "19700101-000000");
        // 2024-02-29T12:34:56Z == 1709210096 (leap day).
        assert_eq!(utc_timestamp_compact(1_709_210_096), "20240229-123456");
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn cpu_and_memory_probes_work_on_linux() {
        assert!(process_cpu_seconds().is_some());
        assert!(peak_rss_bytes().unwrap() > 0);
    }
}
