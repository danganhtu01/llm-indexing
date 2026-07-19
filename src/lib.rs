pub mod config;
pub mod embedding;
pub mod extract;
pub mod media;
pub mod model;
pub mod normalize;
pub mod ocr;
pub mod pipeline;
pub mod service;
pub mod store;
pub mod vision;
pub mod walker;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
