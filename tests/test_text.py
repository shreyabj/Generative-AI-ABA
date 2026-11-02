"""Script-aware text utilities.

These functions underpin chunking, BM25 and every guardrail, so a regression here is a
regression everywhere. The Indic cases are the point: an English-only test suite would pass
happily while the tokeniser silently dropped Devanagari.
"""

from __future__ import annotations

import pytest
from setu.text import (
    containment,
    content_tokens,
    detect_lang,
    detect_script,
    normalize,
    numbers_in,
    split_sentences,
    tokenize,
    truncate,
)


class TestScriptDetection:
    @pytest.mark.parametrize(
        "text,script",
        [
            ("what is a corporation", "latin"),
            ("एक केले में कितनी कैलोरी होती है", "devanagari"),
            ("একটি কলায় কত ক্যালরি আছে", "bengali"),
            ("ஒரு வாழைப்பழத்தில் எத்தனை கலோரிகள்", "tamil"),
            ("ఒక అరటిపండులో ఎన్ని కేలరీలు", "telugu"),
            ("ಒಂದು ಬಾಳೆಹಣ್ಣಿನಲ್ಲಿ ಎಷ್ಟು ಕ್ಯಾಲರಿ", "kannada"),
            ("ഒരു വാഴപ്പഴത്തിൽ എത്ര കലോറി", "malayalam"),
            ("ایک کیلے میں کتنی کیلوریز ہیں", "arabic"),
        ],
    )
    def test_detects_script(self, text, script):
        assert detect_script(text) == script

    def test_mixed_script_picks_dominant(self):
        # A stray English word must not flip a Hindi query to Latin.
        assert detect_script("भारत की GDP कितनी है") == "devanagari"

    def test_empty_defaults_to_latin(self):
        assert detect_script("") == "latin"
        assert detect_script("12345 !!!") == "latin"

    def test_explicit_hint_wins_over_script(self):
        # Devanagari is shared by hi/mr/ne/sa, so the STT language code must override.
        assert detect_lang("नमस्ते", "mar") == "mr"
        assert detect_lang("नमस्ते", "ne") == "ne"
        assert detect_lang("नमस्ते") == "hi"  # most common member as the default


class TestTokenize:
    def test_latin(self):
        assert tokenize("What is a Corporation?") == ["what", "is", "a", "corporation"]

    def test_keeps_numbers_and_decimals(self):
        assert "89" in tokenize("about 89 calories")
        assert "3.5" in tokenize("3.5 million people")

    def test_indic_scripts_are_not_dropped(self):
        # The critical case: a naive [a-z]+ pattern returns [] here.
        for text in ("एक केले में कितनी कैलोरी", "ஒரு வாழைப்பழம்", "একটি কলা"):
            assert len(tokenize(text)) >= 2, text

    def test_content_tokens_drop_stopwords(self):
        toks = content_tokens("what is the meaning of a corporation")
        assert "corporation" in toks
        assert "the" not in toks and "is" not in toks

    def test_content_tokens_drop_indic_stopwords(self):
        toks = content_tokens("कॉर्पोरेशन क्या है")
        assert "है" not in toks


class TestSentences:
    def test_splits_latin(self):
        s = split_sentences("First one. Second one! Third one?")
        assert len(s) == 3

    def test_splits_devanagari_danda(self):
        s = split_sentences("पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।")
        assert len(s) == 3

    def test_offsets_are_exact(self):
        text = "Alpha sentence. Beta sentence. Gamma sentence."
        for sent in split_sentences(text):
            # The recorded span must slice back to the sentence: chunk char_start /
            # char_end and all parent expansion depend on this being exact.
            assert text[sent.start : sent.end] == sent.text

    def test_abbreviation_does_not_split(self):
        s = split_sentences("Dr. Rao founded the company. It grew quickly.")
        assert len(s) == 2

    def test_single_sentence_and_empty(self):
        assert len(split_sentences("no terminator here")) == 1
        assert split_sentences("") == []


class TestNumbers:
    def test_extracts_and_normalises(self):
        assert numbers_in("about 89 calories") == ["89"]
        assert numbers_in("1,234 people") == ["1234"]
        assert numbers_in("15%") == ["15"]

    def test_normalisation_makes_formats_comparable(self):
        # This is what lets the hallucination check compare "1,234" to "1234".
        assert set(numbers_in("1,234")) == set(numbers_in("1234"))

    def test_no_numbers(self):
        assert numbers_in("no digits at all") == []


class TestContainment:
    def test_asymmetric(self):
        claim = {"banana", "89", "calories"}
        evidence = {"a", "banana", "contains", "89", "calories", "per", "100", "grams"}
        assert containment(claim, evidence) == 1.0
        # The reverse is deliberately not 1.0 - a short claim citing a long passage is
        # the normal case, which is why groundedness uses containment and not Jaccard.
        assert containment(evidence, claim) < 0.5

    def test_empty_claim_is_vacuously_supported(self):
        assert containment(set(), {"anything"}) == 1.0


class TestTruncate:
    def test_short_text_untouched(self):
        assert truncate("short", 50) == "short"

    def test_prefers_sentence_boundary(self):
        assert truncate("First sentence. Second sentence follows here.", 24) == "First sentence."

    def test_never_cuts_mid_word(self):
        out = truncate("alpha beta gamma delta epsilon zeta", 20)
        assert not out.rstrip("\u2026").endswith(("g", "d", "e")) or " " in out
        for word in out.rstrip("\u2026").split():
            assert word in "alpha beta gamma delta epsilon zeta".split()

    def test_devanagari_not_corrupted(self):
        # The regression this was written for: a naive character cut landed inside a
        # word and produced text that read as corrupted rather than abbreviated.
        text = "एक केले में प्रति सौ ग्राम लगभग नवासी कैलोरी होती है और यह पोटेशियम का स्रोत है"
        out = truncate(text, 40)
        assert len(out) <= 42
        for word in out.rstrip("\u2026").split():
            assert word in text.split()


class TestNormalize:
    def test_collapses_whitespace(self):
        assert normalize("  too   many\n\nspaces  ", casefold=False) == "too many spaces"

    def test_strips_zero_width(self):
        assert "\u200b" not in normalize("a\u200bb")

    def test_casefold_optional(self):
        assert normalize("MiXeD", casefold=False) == "MiXeD"
        assert normalize("MiXeD") == "mixed"
