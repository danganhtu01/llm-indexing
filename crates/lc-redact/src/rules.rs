//! Redaction rules, and the spans they match.
//!
//! Every rule returns *spans* rather than a rewritten string. That is the whole design:
//! a compliance pipeline has to be able to answer "what did you remove, from where, and
//! under which rule" long after the fact. Rewriting in place throws that away.
//!
//! # Why not just use claudeops' `redact` crate
//!
//! Its rule set is good, but for a *document* corpus it has four gaps, and three of them
//! are silent — they fail by leaving PII in place, not by erroring:
//!
//! 1. **No person names.** It is pure regex with no NER and no gazetteer.
//! 2. **MST (tax code) only as a JSON key name.** No free-text pattern at all.
//! 3. **Context-gated national IDs and bank accounts.** A bare 12-digit CCCD is left
//!    alone unless a keyword sits next to it. In a JSON event that is a reasonable
//!    trade. In a scanned table, the keyword is a column header three rows up, and the
//!    number survives.
//! 4. **No audit trail.** It records nothing about what it changed.
//!
//! So the context-gated rules here run in two modes, and the strict one is the default.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// What a span was matched as. Kept coarse on purpose: this lands in the audit log, and
/// the audit log must not become a second copy of the PII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Email,
    PhoneVn,
    NationalId,
    TaxCode,
    BankAccount,
    PersonName,
}

/// One redacted region of one file.
///
/// Serialize-only: `rule` is a `&'static str` pointing at a rule name baked into the
/// binary, which is exactly what we want in the audit log and impossible to deserialize
/// back into. Read `redactions.jsonl` with `serde_json::Value` if you need to.
#[derive(Debug, Clone, Serialize)]
pub struct Span {
    /// Byte offset into the *original* text.
    pub start: usize,
    pub end: usize,
    pub kind: Kind,
    /// Which rule fired. Stable string; the UI groups by it.
    pub rule: &'static str,
    /// Salted SHA-256 of the matched text, truncated. Lets you prove the same value
    /// appeared in two documents **without storing the value**. Never reversible without
    /// the salt, and the salt is not persisted.
    pub digest: String,
}

/// How aggressive to be about identifiers that look like PII but carry no nearby keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    /// Redact a bare 9/12-digit run or bank-account-shaped number even with no keyword
    /// beside it. Over-redacts invoice numbers; never leaks a national ID. **Default.**
    Strict,
    /// Require an adjacent keyword, as claudeops' `redact` does. Fewer false positives,
    /// but a number in a table cell whose header is three rows up survives.
    ContextGated,
}

pub const REDACTED: &str = "[REDACTED]";

// --------------------------------------------------------------------------------------
// Patterns
// --------------------------------------------------------------------------------------

static EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?-u:\b)[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?-u:\b)").unwrap()
});

/// Vietnamese mobile/landline: +84 or leading 0, then 9 digits, tolerating separators.
static PHONE_VN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\+?84|0)(?:[\s.\-]?[0-9]){9}(?-u:\b)").unwrap());

/// CCCD (12 digits) or the older CMND (9 digits). Bare — see [`Gating`].
static NATIONAL_ID_BARE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?-u:\b)(?:[0-9]{12}|[0-9]{9})(?-u:\b)").unwrap());

/// Same, but only when a keyword precedes it. Capture group 2 is the number.
static NATIONAL_ID_GATED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)((?:cccd|cmnd|cmt|cmtnd|national[\s._-]?id|s(?:ố|o)[\s._-]?cccd|s(?:ố|o)[\s._-]?cmnd)[\s:.\-]*)([0-9]{9}(?:[0-9]{3})?)",
    )
    .unwrap()
});

/// Vietnamese tax code (MST): 10 digits, optionally `-` plus a 3-digit branch suffix.
/// claudeops has no free-text pattern for this at all.
static TAX_CODE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?-u:\b)[0-9]{10}(?:-[0-9]{3})?(?-u:\b)").unwrap());

/// Bank account: 6–24 digits with optional separators, keyword-gated.
static BANK_GATED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)((?:account|acct|a/c|stk|s(?:ố|o)[\s._-]?tk|iban|bank)[\s:.\-]*)([0-9][0-9\s.\-]{5,23}[0-9])",
    )
    .unwrap()
});

// --------------------------------------------------------------------------------------
// Person names
// --------------------------------------------------------------------------------------

/// Person names cannot be done with a regex, and claudeops does not try.
///
/// A gazetteer is the honest 80% answer for Vietnamese: the surname space is tiny
/// (~14 names cover most of the population) and highly distinctive, so
/// `<Surname> <Given> <Given>` in title case is a strong signal. It over-matches on
/// place names that share a surname (Nguyễn Huệ is a street). It under-matches on
/// foreign names entirely.
///
/// Load with `--names <file>` (one per line) to extend. This is a *deliberate* 80%
/// solution, and `run.json` records how many names came from the gazetteer so nobody
/// mistakes it for NER.
pub const VN_SURNAMES: &[&str] = &[
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ",
    "Hồ", "Ngô", "Dương", "Lý", "Đào", "Đinh", "Trịnh", "Mai",
];

/// `Surname Given [Given]` in title case, where Surname is in the gazetteer.
fn person_name_regex(surnames: &[String]) -> Regex {
    let alt = surnames
        .iter()
        .map(|s| regex::escape(s))
        .collect::<Vec<_>>()
        .join("|");
    // \p{Lu}\p{Ll}+ = a title-cased word, Unicode-aware so Vietnamese diacritics work.
    Regex::new(&format!(r"(?:{alt})(?:\s+\p{{Lu}}\p{{Ll}}+){{1,3}}")).expect("person-name regex")
}

// --------------------------------------------------------------------------------------
// Engine
// --------------------------------------------------------------------------------------

pub struct Redactor {
    gating: Gating,
    person: Option<Regex>,
    salt: [u8; 16],
}

impl Redactor {
    pub fn new(gating: Gating, extra_surnames: &[String], salt: [u8; 16]) -> Self {
        let mut names: Vec<String> = VN_SURNAMES.iter().map(|s| s.to_string()).collect();
        names.extend(extra_surnames.iter().cloned());
        // Longest first, so "Nguyễn" cannot be shadowed by a shorter prefix.
        names.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        Self {
            gating,
            person: Some(person_name_regex(&names)),
            salt,
        }
    }

    /// Find every span to redact, sorted by start offset, with overlaps resolved.
    pub fn find(&self, text: &str) -> Vec<Span> {
        let mut spans: Vec<Span> = Vec::new();

        for m in EMAIL.find_iter(text) {
            spans.push(self.span(m.start(), m.end(), Kind::Email, "email", text));
        }
        for m in PHONE_VN.find_iter(text) {
            spans.push(self.span(m.start(), m.end(), Kind::PhoneVn, "phone_vn", text));
        }
        for m in TAX_CODE.find_iter(text) {
            spans.push(self.span(m.start(), m.end(), Kind::TaxCode, "tax_code_mst", text));
        }

        match self.gating {
            Gating::Strict => {
                for m in NATIONAL_ID_BARE.find_iter(text) {
                    spans.push(self.span(
                        m.start(),
                        m.end(),
                        Kind::NationalId,
                        "national_id_bare",
                        text,
                    ));
                }
            }
            Gating::ContextGated => {
                for c in NATIONAL_ID_GATED.captures_iter(text) {
                    // Group 2 is the number; the keyword is kept, as claudeops does.
                    let m = c.get(2).expect("group 2");
                    spans.push(self.span(
                        m.start(),
                        m.end(),
                        Kind::NationalId,
                        "national_id_gated",
                        text,
                    ));
                }
            }
        }

        // Bank accounts stay keyword-gated in both modes: an ungated 6–24-digit run
        // matches almost every invoice line, and the false-positive rate destroys the
        // corpus rather than protecting it.
        for c in BANK_GATED.captures_iter(text) {
            let m = c.get(2).expect("group 2");
            spans.push(self.span(m.start(), m.end(), Kind::BankAccount, "bank_gated", text));
        }

        if let Some(re) = &self.person {
            for m in re.find_iter(text) {
                spans.push(self.span(
                    m.start(),
                    m.end(),
                    Kind::PersonName,
                    "person_name_gazetteer",
                    text,
                ));
            }
        }

        resolve_overlaps(spans)
    }

    /// Apply spans, returning the redacted text. Offsets refer to the original.
    pub fn apply(&self, text: &str, spans: &[Span]) -> String {
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for s in spans {
            // Spans are byte offsets from a regex over this same &str, so they are always
            // on char boundaries. Guard anyway: a panic here would kill a whole run.
            if s.start < cursor || s.end > text.len() || !text.is_char_boundary(s.start) {
                continue;
            }
            out.push_str(&text[cursor..s.start]);
            out.push_str(REDACTED);
            cursor = s.end;
        }
        out.push_str(&text[cursor..]);
        out
    }

    fn span(&self, start: usize, end: usize, kind: Kind, rule: &'static str, text: &str) -> Span {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.salt);
        h.update(&text.as_bytes()[start..end]);
        let digest = h.finalize();
        Span {
            start,
            end,
            kind,
            rule,
            digest: hex16(&digest[..8]),
        }
    }
}

fn hex16(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Sort by start, then drop any span contained in or overlapping an earlier one.
///
/// Overlaps are real: a 12-digit CCCD is also a 10-digit tax code plus two digits, and
/// `0912345678` is both a phone number and a 10-digit MST. First match wins after sorting
/// by (start, longest), which prefers the more specific, longer interpretation.
fn resolve_overlaps(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    let mut out: Vec<Span> = Vec::with_capacity(spans.len());
    for s in spans {
        match out.last() {
            Some(prev) if s.start < prev.end => continue,
            _ => out.push(s),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(gating: Gating) -> Redactor {
        Redactor::new(gating, &[], [7u8; 16])
    }

    fn redact(text: &str, gating: Gating) -> String {
        let red = r(gating);
        let spans = red.find(text);
        red.apply(text, &spans)
    }

    #[test]
    fn email_and_phone() {
        let out = redact("mail a@b.com or ring 0912345678 today", Gating::Strict);
        assert!(!out.contains("a@b.com"));
        assert!(!out.contains("0912345678"));
    }

    #[test]
    fn bare_national_id_survives_gated_mode_but_not_strict() {
        // This is precisely the claudeops behaviour we are choosing to override.
        let text = "Số: 001199004455 cấp ngày 1/1/2020";
        assert!(redact(text, Gating::ContextGated).contains("001199004455"));
        assert!(!redact(text, Gating::Strict).contains("001199004455"));
    }

    #[test]
    fn gated_mode_keeps_the_keyword_drops_the_number() {
        let out = redact("CCCD: 001199004455", Gating::ContextGated);
        assert!(out.contains("CCCD"));
        assert!(!out.contains("001199004455"));
    }

    #[test]
    fn tax_code_in_free_text() {
        // claudeops catches MST only as a JSON key name; this is the gap.
        let out = redact("Mã số thuế 0101243150-001 của công ty", Gating::Strict);
        assert!(!out.contains("0101243150"));
    }

    #[test]
    fn bank_account_needs_a_keyword_in_both_modes() {
        assert!(!redact("STK 19001234567890", Gating::Strict).contains("19001234567890"));
        // A bare run is left alone: ungating this destroys invoice tables.
        assert!(redact("Invoice 19001234567890", Gating::Strict).contains("19001234567890"));
    }

    #[test]
    fn person_name_gazetteer() {
        let out = redact("Ký bởi Nguyễn Văn An, giám đốc", Gating::Strict);
        assert!(!out.contains("Nguyễn Văn An"));
        assert!(out.contains("giám đốc"));
    }

    #[test]
    fn overlapping_spans_do_not_double_redact() {
        let red = r(Gating::Strict);
        let text = "0912345678";
        let spans = red.find(text);
        // phone (10 chars) and tax code (10 digits) both match the whole string.
        assert_eq!(spans.len(), 1, "overlaps must collapse to one span");
        assert_eq!(red.apply(text, &spans), REDACTED);
    }

    #[test]
    fn digest_is_stable_and_does_not_leak_the_value() {
        let red = r(Gating::Strict);
        let a = red.find("a@b.com");
        let b = red.find("a@b.com");
        assert_eq!(a[0].digest, b[0].digest);
        assert!(!a[0].digest.contains("a@b"));
        assert_eq!(a[0].digest.len(), 16);
    }

    #[test]
    fn clean_text_is_returned_unchanged() {
        let text = "Điều 5. Hợp đồng có hiệu lực kể từ ngày ký.";
        assert_eq!(redact(text, Gating::Strict), text);
    }
}
