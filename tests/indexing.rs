use std::fs;
use std::path::PathBuf;

use llm_indexing::config::Config;
use llm_indexing::normalize::Normalizer;
use llm_indexing::pipeline::{run_index, IndexRequest};
use llm_indexing::store::{connect, search, top_folders};

#[test]
fn indexes_and_searches_english_and_vietnamese() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(input.join("sub")).unwrap();
    fs::write(
        input.join("report_en.txt"),
        "Anti money laundering compliance report. Suspicious activity detected.",
    )
    .unwrap();
    fs::write(
        input.join("bao_cao_vi.txt"),
        "Báo cáo giao dịch đáng ngờ tại ngân hàng. Khách hàng rủi ro cao.",
    )
    .unwrap();
    fs::write(
        input.join("sub/notes.md"),
        "KYC and CDD notes for bank review.",
    )
    .unwrap();

    let mut config = Config::default();
    config.ocr = "off".into();
    config.sidecar = "none".into();
    config.workers = 2;
    config.data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let stats = run_index(IndexRequest {
        paths: std::slice::from_ref(&input),
        out: &output,
        config: config.clone(),
        resume: false,
        artifacts: true,
    })
    .unwrap();
    assert_eq!(stats.files, 3);

    let connection = connect(&output).unwrap();
    let normalizer = Normalizer::load(&config);
    assert!(!search(&connection, &normalizer, "launder", 5, false)
        .unwrap()
        .is_empty());
    assert!(!search(&connection, &normalizer, "ngan hang", 5, false)
        .unwrap()
        .is_empty());
    assert!(
        !search(&connection, &normalizer, "know your customer", 5, false)
            .unwrap()
            .is_empty()
    );
    assert!(!top_folders(&connection, &normalizer, "bank", 5)
        .unwrap()
        .is_empty());
}
