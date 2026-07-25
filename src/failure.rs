//! A small, fixed taxonomy for extraction failures.
//!
//! [`pipeline::process_file`](crate::pipeline) used to derive the stored
//! `error:<token>` method string with `error.root_cause().to_string()
//! .split_whitespace().next()` — the first word of the ROOT CAUSE's `Display`.
//! For a [`std::io::Error`] built from an OS error code, that `Display` is
//! rendered by the OS in the process's locale: on a German Windows box, "The
//! system cannot find the file specified." becomes "Das System kann die
//! angegebene Datei nicht finden.", and the live corpus has 85 rows with
//! `method = 'error:Das'` to show for it. Locale, not failure mode, decided
//! the taxonomy.
//!
//! [`classify`] replaces that: every branch matches on error TYPE or, for
//! `std::io::Error`, on [`std::io::ErrorKind`] — never on `to_string()`. The
//! result is a fixed, stable, locale-independent identifier set.
use std::fmt;

/// The failure classes a stored `error:<class>` method string can take.
///
/// This is deliberately small and closed: adding a class means adding a
/// variant here plus a match arm in [`classify`], not inventing a new prefix
/// somewhere a resume predicate has to learn about. Every existing consumer of
/// `method` (the `error:` prefix check in [`crate::pipeline::incomplete_method`]
/// and the `method LIKE 'error:%'` resume predicate in `store.rs`) matches on
/// the prefix alone, so it treats old locale-derived rows (`error:Das`,
/// `error:local`, `error:program`) and these new stable ones identically —
/// nothing about resume needed to change for this taxonomy to land.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureClass {
    /// `std::io::Error` with kind [`std::io::ErrorKind::NotFound`]: a missing
    /// file, a vanished path mid-walk, or a missing external binary — spawning
    /// a `Command` for an executable that isn't on `PATH` surfaces the same
    /// `NotFound` kind as a missing input file.
    IoNotFound,
    /// `std::io::Error` with kind [`std::io::ErrorKind::PermissionDenied`]: an
    /// ACL, a file exclusively locked by another process, a directory without
    /// read access.
    IoDenied,
    /// The bytes were reachable but not well-formed for their claimed format:
    /// a corrupt/truncated Zip-based container (`.docx`/`.xlsx`/`.pptx`/an
    /// archive extension), a malformed `.eml`, or WAV data `hound` rejects
    /// after the ffmpeg extraction step.
    Decode,
    /// `std::io::Error` with kind [`std::io::ErrorKind::TimedOut`]. Nothing in
    /// this build's extraction path currently produces this — every
    /// subprocess call blocks on `Command::output()` without a deadline — so
    /// this class is presently unreachable, but classification is already
    /// wired to it the moment one gains a bounded wait.
    Timeout,
    /// The document requires a password. Constructed as [`EncryptedDocument`]
    /// by `extract.rs`'s PDF fast path (`pdf()`/`pdf_exhaustive()`) when
    /// `pdfinfo` cannot open the file with poppler's default empty
    /// password — see [`EncryptedDocument`]'s doc comment. A zip-based
    /// Office document that IS already password-protected surfaces this
    /// through a different route, `zip::result::ZipError::InvalidPassword`,
    /// which `classify` also maps here — that is the existing
    /// `office_archive` path noticing a password.
    Encrypted,
    /// A capability this build or its configuration lacks: no Tesseract
    /// binary for exhaustive PDF OCR, no local Whisper model file. See
    /// [`CapabilityUnavailable`].
    Unsupported,
    /// Anything that does not downcast to one of the above: an
    /// `anyhow::bail!` call site with free-form text and no distinguishing
    /// error type (a failed subprocess's exit status, an exhausted safety
    /// limit), or an `io::Error` kind none of the classes above claim.
    Unknown,
}

impl FailureClass {
    /// The stored `error:<token>` suffix. Stable across releases: corpus rows
    /// key resume decisions and any future UI grouping on this string, so it
    /// must not change once shipped.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IoNotFound => "io-not-found",
            Self::IoDenied => "io-denied",
            Self::Decode => "decode",
            Self::Timeout => "timeout",
            Self::Encrypted => "encrypted",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FailureClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A capability this build or its configuration lacks that a format needs in
/// order to extract: no Tesseract binary for exhaustive PDF OCR, no local
/// Whisper model file on disk.
///
/// Built at the call site instead of `anyhow::bail!`-ing a plain string so
/// [`classify`] can recognize the failure by TYPE. The message is kept (and
/// still reaches logs via `Display`/`{error:#}`) purely for humans; nothing
/// classifies on it.
#[derive(Debug)]
pub struct CapabilityUnavailable(pub &'static str);

impl fmt::Display for CapabilityUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for CapabilityUnavailable {}

/// Marker for a PDF that requires a password to open.
///
/// Built by `extract.rs`'s `pdf_info()` (via `pdf()` and `pdf_exhaustive()`)
/// when `pdfinfo` cannot open the document with poppler's default empty user
/// password — bailing with this type instead of letting the file fall
/// through to pdftotext/pdftoppm and an empty `pdf-text-partial` row.
/// [`classify`] downcasts to this type and returns
/// [`FailureClass::Encrypted`] for it; nothing further was needed on this
/// side for B2 to land. An owner-password-only PDF (readable under the
/// default empty user password) never constructs this — see `pdf_info()`'s
/// doc comment for how the two are told apart.
#[derive(Debug)]
pub struct EncryptedDocument;

impl fmt::Display for EncryptedDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("document requires a password")
    }
}

impl std::error::Error for EncryptedDocument {}

/// Map an `std::io::Error`'s kind to a [`FailureClass`]. Shared by the
/// top-of-chain check in [`classify`] and by the library error types below
/// that carry their own `io::Error` (`zip::result::ZipError::Io`,
/// `hound::Error::IoError`) so an io failure is classified the same way
/// whichever format library it surfaced through.
fn classify_io(kind: std::io::ErrorKind) -> FailureClass {
    match kind {
        std::io::ErrorKind::NotFound => FailureClass::IoNotFound,
        std::io::ErrorKind::PermissionDenied => FailureClass::IoDenied,
        std::io::ErrorKind::TimedOut => FailureClass::Timeout,
        _ => FailureClass::Unknown,
    }
}

/// Classify an extraction failure into a [`FailureClass`] by walking the
/// error's source chain and matching on TYPE — never on `to_string()`. See
/// the module doc for why a `Display`-based classifier is the wrong tool: a
/// `std::io::Error`'s `Display` is OS-localized.
///
/// `error.chain()` yields the error itself followed by every `source()` in
/// turn, so this finds an `io::Error` whether it is the direct cause (a bare
/// `File::open(..)?`) or nested inside a library error that wraps one
/// (`zip::result::ZipError::Io`, `hound::Error::IoError`) — in the latter
/// case the wrapping type is matched first so its embedded `io::Error` can be
/// classified by kind via the same [`classify_io`] rather than falling
/// through to [`FailureClass::Decode`].
pub fn classify(error: &anyhow::Error) -> FailureClass {
    for cause in error.chain() {
        if let Some(io_error) = cause.downcast_ref::<std::io::Error>() {
            return classify_io(io_error.kind());
        }
        if cause.downcast_ref::<EncryptedDocument>().is_some() {
            return FailureClass::Encrypted;
        }
        if cause.downcast_ref::<CapabilityUnavailable>().is_some() {
            return FailureClass::Unsupported;
        }
        match cause.downcast_ref::<zip::result::ZipError>() {
            Some(zip::result::ZipError::Io(io_error)) => return classify_io(io_error.kind()),
            Some(zip::result::ZipError::InvalidPassword) => return FailureClass::Encrypted,
            Some(_) => return FailureClass::Decode,
            None => {}
        }
        match cause.downcast_ref::<hound::Error>() {
            Some(hound::Error::IoError(io_error)) => return classify_io(io_error.kind()),
            Some(_) => return FailureClass::Decode,
            None => {}
        }
        if cause.downcast_ref::<mailparse::MailParseError>().is_some() {
            return FailureClass::Decode;
        }
    }
    FailureClass::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect this module fixes: a localized `io::Error` must classify by
    /// KIND, not by its OS-rendered message text. This is the literal shape of
    /// the live corpus's `error:Das` rows — an `ErrorKind::NotFound` whose
    /// `Display` happens to be German.
    #[test]
    fn a_localized_not_found_error_classifies_by_kind_not_text() {
        let german = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Das System kann die angegebene Datei nicht finden. (os error 3)",
        );
        let error: anyhow::Error = anyhow::Error::new(german).context("reading file");
        assert_eq!(classify(&error), FailureClass::IoNotFound);
        assert_eq!(classify(&error).as_str(), "io-not-found");
    }

    #[test]
    fn permission_denied_classifies_regardless_of_locale() {
        let localized = std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "L'accès est refusé. (os error 5)",
        );
        let error: anyhow::Error = anyhow::Error::new(localized);
        assert_eq!(classify(&error), FailureClass::IoDenied);
    }

    #[test]
    fn a_missing_external_binary_is_io_not_found() {
        // What `Command::new("antiword").output()` returns when the binary
        // isn't on PATH: an `io::Error` with kind `NotFound`, same as a
        // missing input file — a real source of the same class, not a
        // synthetic one.
        let spawn_failure = std::process::Command::new("this-binary-does-not-exist-anywhere")
            .output()
            .unwrap_err();
        assert_eq!(spawn_failure.kind(), std::io::ErrorKind::NotFound);
        let error: anyhow::Error =
            anyhow::Error::new(spawn_failure).context("running antiword for a.doc");
        assert_eq!(classify(&error), FailureClass::IoNotFound);
    }

    #[test]
    fn a_timed_out_io_error_classifies_as_timeout() {
        let error: anyhow::Error = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "operation timed out",
        ));
        assert_eq!(classify(&error), FailureClass::Timeout);
    }

    #[test]
    fn an_unmapped_io_kind_falls_back_to_unknown() {
        let error: anyhow::Error = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "already exists",
        ));
        assert_eq!(classify(&error), FailureClass::Unknown);
    }

    #[test]
    fn a_corrupt_zip_container_is_decode() {
        let error: anyhow::Error =
            anyhow::Error::new(zip::result::ZipError::InvalidArchive("bad local header"));
        assert_eq!(classify(&error), FailureClass::Decode);
    }

    #[test]
    fn an_io_error_wrapped_by_zip_still_classifies_by_kind() {
        let wrapped = zip::result::ZipError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Zugriff verweigert",
        ));
        let error: anyhow::Error = anyhow::Error::new(wrapped);
        assert_eq!(classify(&error), FailureClass::IoDenied);
    }

    #[test]
    fn a_password_protected_zip_container_is_encrypted() {
        let error: anyhow::Error = anyhow::Error::new(zip::result::ZipError::InvalidPassword);
        assert_eq!(classify(&error), FailureClass::Encrypted);
    }

    #[test]
    fn a_malformed_email_is_decode() {
        let error: anyhow::Error =
            anyhow::Error::new(mailparse::MailParseError::Generic("bad header"));
        assert_eq!(classify(&error), FailureClass::Decode);
    }

    #[test]
    fn malformed_wav_data_is_decode() {
        let error: anyhow::Error = anyhow::Error::new(hound::Error::FormatError("not a WAVE file"));
        assert_eq!(classify(&error), FailureClass::Decode);
    }

    #[test]
    fn an_io_error_wrapped_by_hound_still_classifies_by_kind() {
        let wrapped = hound::Error::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "extracted WAV vanished",
        ));
        let error: anyhow::Error = anyhow::Error::new(wrapped);
        assert_eq!(classify(&error), FailureClass::IoNotFound);
    }

    #[test]
    fn a_missing_capability_is_unsupported() {
        let error: anyhow::Error =
            anyhow::Error::new(CapabilityUnavailable("Tesseract is unavailable"));
        assert_eq!(classify(&error), FailureClass::Unsupported);
    }

    /// `extract.rs`'s PDF fast path constructs this directly; this test
    /// covers the classification side in isolation from the subprocess
    /// plumbing.
    #[test]
    fn the_reserved_encrypted_marker_classifies_as_encrypted() {
        let error: anyhow::Error = anyhow::Error::new(EncryptedDocument);
        assert_eq!(classify(&error), FailureClass::Encrypted);
    }

    #[test]
    fn a_freeform_bail_with_no_distinguishing_type_is_unknown() {
        let error = anyhow::anyhow!("failed to rasterize PDF page 3");
        assert_eq!(classify(&error), FailureClass::Unknown);
    }

    /// A message-based classifier would key off "The" (or whatever locale
    /// renders first); the type-based one must give the SAME answer whatever
    /// the wording is. This is the regression test for the bug this module
    /// exists to fix.
    #[test]
    fn classification_is_invariant_to_the_error_message_text() {
        for message in [
            "The system cannot find the file specified. (os error 2)",
            "Das System kann die angegebene Datei nicht finden. (os error 2)",
            "Le fichier spécifié est introuvable. (os error 2)",
        ] {
            let error: anyhow::Error =
                anyhow::Error::new(std::io::Error::new(std::io::ErrorKind::NotFound, message));
            assert_eq!(classify(&error), FailureClass::IoNotFound, "{message}");
        }
    }
}
