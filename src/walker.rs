use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::config::Config;
use crate::model::FileRec;

pub fn walk(paths: &[PathBuf], config: &Config) -> Vec<FileRec> {
    let mut files = Vec::new();
    for requested in paths {
        let root = requested
            .canonicalize()
            .unwrap_or_else(|_| requested.clone());
        let drive = drive_label(&root);
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_symlink() && !config.follow_symlinks {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if file_type.is_dir() {
                    if !config.skip_dir(&name) {
                        stack.push(path)
                    }
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }
                let Ok(meta) = entry.metadata() else { continue };
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or_default();
                let ext = path
                    .extension()
                    .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
                    .unwrap_or_default();
                files.push(FileRec {
                    path: path.to_string_lossy().to_string(),
                    name,
                    ext,
                    dir: path
                        .parent()
                        .unwrap_or(Path::new(""))
                        .to_string_lossy()
                        .to_string(),
                    drive: drive.clone(),
                    size: meta.len(),
                    mtime,
                });
            }
        }
    }
    files
}

fn drive_label(path: &Path) -> String {
    #[cfg(windows)]
    {
        use std::path::Component;
        if let Some(Component::Prefix(prefix)) = path.components().next() {
            return prefix.as_os_str().to_string_lossy().to_string();
        }
    }
    path.components()
        .next()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .unwrap_or_else(|| "/".into())
}
