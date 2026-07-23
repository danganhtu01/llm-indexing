use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::config::Config;
use crate::model::FileRec;

/// The canonical form of a requested root — the SAME resolution [`walk`]
/// applies to every root before descending, kept as one function so path
/// strings derived from a root elsewhere (the resume prune's root scoping)
/// can never drift from what the walker actually wrote into the corpus.
pub fn canonical_root(requested: &Path) -> PathBuf {
    requested
        .canonicalize()
        .unwrap_or_else(|_| requested.to_path_buf())
}

pub fn walk(paths: &[PathBuf], config: &Config) -> Vec<FileRec> {
    let mut files = Vec::new();
    for requested in paths {
        let root = canonical_root(requested);
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
                    // Two independent prunes: `skip_dir` on the bare basename
                    // (node_modules anywhere), `skip_path` on the anchored full
                    // path (C:\Windows but not D:\projects\Windows). A match on
                    // either means the directory is never descended into.
                    if !config.skip_dir(&name) && !config.skip_path(&path) {
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
        use std::path::{Component, Prefix};
        if let Some(Component::Prefix(prefix)) = path.components().next() {
            // Read the drive letter off the PARSED prefix kind, never the raw
            // prefix string: production paths arrive `canonicalize`d, i.e.
            // `\\?\`-verbatim (`Prefix::VerbatimDisk`), whose literal prefix is
            // `\\?\I:` — so `as_os_str()` stored the verbatim form instead of the
            // bare `I:` the rest of the codebase keys on (config `skip_path`, the
            // drives-analytics grouping). Disk and VerbatimDisk both normalize to
            // `X:`; a prefix with no disk letter (a UNC share) keeps its literal
            // string, there being no bare-drive form to reduce it to.
            return match prefix.kind() {
                Prefix::Disk(letter) | Prefix::VerbatimDisk(letter) => {
                    format!("{}:", letter as char)
                }
                _ => prefix.as_os_str().to_string_lossy().to_string(),
            };
        }
    }
    path.components()
        .next()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .unwrap_or_else(|| "/".into())
}

#[cfg(test)]
mod tests {
    use super::walk;
    use crate::config::Config;
    use std::fs;
    use std::path::PathBuf;

    /// The drive label is the bare disk prefix (`I:`), never the `\\?\`-verbatim
    /// form. The verbatim cases are the ones production actually stores, since the
    /// walker canonicalizes every root before descending — the old `as_os_str()`
    /// path wrote `\\?\I:` into `files.drive`, which no consumer keying on the bare
    /// `X:` form (config `skip_path`, drive analytics) would match. Windows-only:
    /// off Windows there is no disk prefix to read.
    #[cfg(windows)]
    #[test]
    fn drive_label_normalizes_the_disk_prefix() {
        use super::drive_label;
        use std::path::Path;
        assert_eq!(drive_label(Path::new(r"I:\photos\a.jpg")), "I:");
        assert_eq!(drive_label(Path::new(r"C:\x.txt")), "C:");
        // Verbatim (`\\?\`) form from `canonicalize` must reduce to the bare disk.
        assert_eq!(drive_label(Path::new(r"\\?\I:\photos\a.jpg")), "I:");
        assert_eq!(drive_label(Path::new(r"\\?\C:\x.txt")), "C:");
    }

    /// Defaults as `--config`-less startup produces them, with `skip_paths`
    /// swapped and recompiled.
    fn finalized(skip_paths: Vec<String>) -> Config {
        let mut config = Config::load(None).unwrap();
        config.skip_paths = skip_paths;
        config.finalize();
        config
    }

    fn walked_names(root: &PathBuf, config: &Config) -> Vec<String> {
        walk(std::slice::from_ref(root), config)
            .into_iter()
            .map(|rec| rec.name)
            .collect()
    }

    /// The anchoring fix, proven against the filesystem rather than the matcher.
    /// Windows-only: off Windows "Windows" is deliberately still a bare
    /// `skip_dirs` entry, so this folder stays pruned there.
    #[cfg(windows)]
    #[test]
    fn a_user_folder_named_windows_is_walked_not_pruned() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("projects");
        fs::create_dir_all(root.join("Windows")).unwrap();
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::write(root.join("Windows").join("notes.txt"), "user data").unwrap();
        fs::write(root.join("node_modules").join("pkg.txt"), "junk").unwrap();

        let config = finalized(Config::default().skip_paths);
        let names = walked_names(&root, &config);
        // Real user data under a folder that merely shares the OS name.
        assert!(names.iter().any(|n| n == "notes.txt"), "{names:?}");
        // Genuinely name-based exclusions still prune.
        assert!(!names.iter().any(|n| n == "pkg.txt"), "{names:?}");
    }

    /// The walker consults `skip_path`, and a hit prunes the whole subtree.
    #[test]
    fn a_matching_skip_path_prunes_the_subtree() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        fs::create_dir_all(root.join("excluded").join("nested")).unwrap();
        fs::create_dir_all(root.join("kept")).unwrap();
        fs::write(root.join("excluded").join("nested").join("deep.txt"), "x").unwrap();
        fs::write(root.join("kept").join("keep.txt"), "y").unwrap();

        // Canonicalize first: `walk` canonicalizes every root, and on Windows
        // that both expands 8.3 short names and adds the `\\?\` prefix.
        let canonical = root.canonicalize().unwrap();
        let pattern = canonical.join("excluded").to_string_lossy().to_string();
        let config = finalized(vec![pattern]);

        let names = walked_names(&root, &config);
        assert!(!names.iter().any(|n| n == "deep.txt"), "{names:?}");
        assert!(names.iter().any(|n| n == "keep.txt"), "{names:?}");
    }

    /// An explicitly requested root is never itself tested against `skip_paths`,
    /// so pointing the indexer at an excluded location still works. This is why
    /// `PathPattern::matches` is an exact match rather than a prefix match.
    #[test]
    fn an_explicitly_requested_root_is_still_walked() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("sub").join("inside.txt"), "x").unwrap();

        let canonical = root.canonicalize().unwrap();
        let config = finalized(vec![canonical.to_string_lossy().to_string()]);

        let names = walked_names(&root, &config);
        assert!(names.iter().any(|n| n == "inside.txt"), "{names:?}");
    }
}
