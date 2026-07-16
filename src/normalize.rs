use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

use crate::config::Config;

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[\p{L}\p{N}]+").expect("static token regex"))
}

pub fn fold(text: &str) -> String {
    text.nfd()
        .filter(|c| !is_combining_mark(*c))
        .map(|c| match c {
            'đ' => 'd',
            'Đ' => 'D',
            _ => c,
        })
        .collect()
}

pub struct Normalizer {
    en_words: HashSet<String>,
    vi_words: HashSet<String>,
    vi_max_len: usize,
    abbreviations: HashMap<String, Vec<String>>,
    stemmer: Stemmer,
}

impl Normalizer {
    pub fn load(config: &Config) -> Self {
        let dict = config.data_dir.join("dict");
        let en_words = load_dic(&dict.join("en_US.dic"));
        let mut vi_words = load_dic(&dict.join("vi_VN.dic"));
        let vi_compounds = load_lines(&dict.join("vi_words.txt"));
        vi_words.extend(vi_compounds);
        let vi_max_len = vi_words
            .iter()
            .map(|w| w.split_whitespace().count())
            .max()
            .unwrap_or(1)
            .min(6);
        let mut abbreviations = HashMap::new();
        for name in ["abbreviations_en.txt", "abbreviations_vi.txt"] {
            load_abbreviations(&config.data_dir.join(name), &mut abbreviations);
        }
        Self {
            en_words,
            vi_words,
            vi_max_len,
            abbreviations,
            stemmer: Stemmer::create(Algorithm::English),
        }
    }

    pub fn detect_lang(&self, text: &str) -> String {
        let sample: String = text.chars().take(2_000).collect();
        let words = words(&sample);
        if words.is_empty() {
            return "und".into();
        }
        let total_chars = sample.chars().count().max(1) as f64;
        // Cyrillic is unambiguous among the languages we support, so decide it
        // by script before the Latin-script vi/en/de heuristics below.
        let cyrillic = sample.chars().filter(|c| is_cyrillic(*c)).count();
        if cyrillic as f64 / total_chars > 0.02 {
            return "ru".into();
        }
        let vi_marks = sample.chars().filter(|c| is_vietnamese(*c)).count();
        let sampled = words.iter().take(300).collect::<Vec<_>>();
        let en_hits = sampled
            .iter()
            .filter(|w| self.en_words.contains(w.as_str()))
            .count();
        let vi_hits = sampled
            .iter()
            .filter(|w| self.vi_words.contains(w.as_str()))
            .count();
        let de_hits = sampled
            .iter()
            .filter(|w| GERMAN_STOPWORDS.contains(&w.as_str()))
            .count();
        let n = sampled.len().max(1) as f64;
        let is_vi = vi_marks as f64 / total_chars > 0.02 || vi_hits as f64 / n > 0.35;
        let is_en = en_hits as f64 / n > 0.35;
        // German shares the Latin script (and many cognates) with English, so we
        // key off German-specific function words / the ß grapheme rather than a
        // Hunspell dictionary we don't ship; Vietnamese diacritics win outright.
        let is_de = !is_vi && (sample.contains('ß') || de_hits as f64 / n > 0.15);
        match (is_vi, is_en, is_de, vi_marks > 0) {
            (true, true, _, _) => "mixed",
            (true, false, _, _) | (false, false, false, true) => "vi",
            (false, _, true, _) => "de",
            (false, true, false, _) => "en",
            _ if en_hits >= vi_hits && en_hits > 0 => "en",
            _ => "und",
        }
        .into()
    }

    pub fn enrich(&self, text: &str, lang: &str) -> Vec<String> {
        let input = words(text);
        let mut out = input.clone();
        for word in &input {
            let folded = fold(word);
            if folded != *word {
                out.push(folded)
            }
        }
        if matches!(lang, "en" | "mixed" | "und") {
            for word in &input {
                if word.is_ascii() && word.len() > 3 {
                    let stem = self.stemmer.stem(word).to_string();
                    if stem != *word {
                        out.push(stem)
                    }
                }
            }
        }
        if matches!(lang, "vi" | "mixed" | "und") {
            for compound in self.segment_vi(&input) {
                let joined = compound.replace(' ', "_");
                let folded = fold(&joined);
                out.push(joined.clone());
                if folded != joined {
                    out.push(folded)
                }
            }
        }
        for word in &input {
            if let Some(expansions) = self.abbreviations.get(word) {
                for expansion in expansions {
                    out.extend(words(expansion))
                }
            }
        }
        let mut seen = HashSet::new();
        out.into_iter()
            .filter(|token| !token.is_empty() && seen.insert(token.clone()))
            .collect()
    }

    pub fn query_tokens(&self, query: &str) -> Vec<String> {
        self.enrich(query, "und")
    }

    fn segment_vi(&self, input: &[String]) -> Vec<String> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < input.len() {
            let mut chosen = 1;
            for len in (2..=self.vi_max_len.min(input.len() - i)).rev() {
                if self.vi_words.contains(&input[i..i + len].join(" ")) {
                    chosen = len;
                    break;
                }
            }
            if chosen > 1 {
                result.push(input[i..i + chosen].join(" "))
            }
            i += chosen;
        }
        result
    }
}

pub fn words(text: &str) -> Vec<String> {
    token_re()
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn load_dic(path: &Path) -> HashSet<String> {
    load_lines(path)
        .into_iter()
        .map(|line| line.split('/').next().unwrap_or(&line).to_lowercase())
        .filter(|line| line.parse::<usize>().is_err())
        .collect()
}

fn load_lines(path: &Path) -> HashSet<String> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_lowercase)
        .collect()
}

fn load_abbreviations(path: &Path, target: &mut HashMap<String, Vec<String>>) {
    for line in fs::read_to_string(path).unwrap_or_default().lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let pair = line.split_once('\t').or_else(|| line.split_once('='));
        let Some((key, values)) = pair else { continue };
        let bucket = target.entry(key.trim().to_lowercase()).or_default();
        for value in values
            .split([';', ','])
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            if !bucket.iter().any(|existing| existing == value) {
                bucket.push(value.to_lowercase())
            }
        }
    }
}

fn is_vietnamese(c: char) -> bool {
    "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        .contains(c.to_lowercase().next().unwrap_or(c))
}

/// Cyrillic + Cyrillic Supplement blocks — any presence flags Russian, the only
/// Cyrillic-script language we index.
fn is_cyrillic(c: char) -> bool {
    matches!(c, '\u{0400}'..='\u{052F}')
}

/// Common German function words used as a cheap language signal — these rarely
/// occur in English or Vietnamese text, so a handful of hits flags German
/// without pulling in a full dictionary or a langdetect dependency.
const GERMAN_STOPWORDS: &[&str] = &[
    "der", "die", "das", "und", "ist", "nicht", "mit", "ein", "eine", "auch", "für", "von", "den",
    "dem", "des", "sich", "auf", "werden", "wird", "aber", "oder", "als", "zum", "zur", "im",
    "bei", "aus", "dass", "einer", "eines",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    fn normalizer() -> Normalizer {
        // The Hunspell dicts are fetched at build time and absent from a bare
        // checkout, but ru/de detection is script/stopword-based and needs none.
        let mut config = Config::default();
        config.data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
        Normalizer::load(&config)
    }

    #[test]
    fn folds_vietnamese() {
        assert_eq!(fold("Ngân hàng Đặng"), "Ngan hang Dang");
    }

    #[test]
    fn folds_cyrillic_without_deleting_letters() {
        // Stripping combining marks must not drop Cyrillic base letters — only
        // the stress/accent marks (й→и, ё→е) fold away, leaving searchable text.
        assert_eq!(fold("Москва"), "Москва");
        assert_eq!(fold("Ёлка"), "Елка");
    }

    #[test]
    fn folds_german_umlauts_and_eszett() {
        assert_eq!(fold("Müller"), "Muller");
        // ß has no combining-mark decomposition, so it survives folding intact.
        assert_eq!(fold("Straße"), "Straße");
    }

    #[test]
    fn detects_russian_by_script() {
        assert_eq!(normalizer().detect_lang("Москва — столица России."), "ru");
    }

    #[test]
    fn detects_german_by_stopwords() {
        assert_eq!(
            normalizer()
                .detect_lang("Der Vertrag ist mit dem Kunden nicht unterschrieben und storniert."),
            "de"
        );
    }
}
