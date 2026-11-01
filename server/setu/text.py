"""Script-aware text utilities.

MS MARCO-XI spans 14 Indic languages plus English, which means naive
``text.split()`` and English sentence splitters both fall apart. Everything in
this module is written against Unicode blocks and works for Devanagari, Bengali,
Tamil, Telugu, Kannada, Malayalam, Odia, Gurmukhi, Gujarati and Perso-Arabic.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------- script ranges
SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
)

# Default language guess per script. Ambiguous scripts (Devanagari is used by
# Hindi, Marathi, Nepali and Sanskrit) fall back to the most common member and
# are refined by the STT language code when one is available.
SCRIPT_DEFAULT_LANG: dict[str, str] = {
    "devanagari": "hi",
    "bengali": "bn",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "odia": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "arabic": "ur",
    "latin": "en",
}

LANG_NAMES: dict[str, str] = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ne": "Nepali",
    "sa": "Sanskrit", "ur": "Urdu",
}

# ai4bharat file-stem -> ISO-639-1
XI_LANGS: dict[str, str] = {
    "asm": "as", "ben": "bn", "guj": "gu", "hin": "hi", "kan": "kn", "mal": "ml",
    "mar": "mr", "nep": "ne", "ori": "or", "pan": "pa", "san": "sa", "tam": "ta",
    "tel": "te", "urd": "ur",
}

# ISO-639-1 -> ISO-639-3 (what the ElevenLabs STT language hint wants)
ISO1_TO_ISO3: dict[str, str] = {
    "en": "eng", "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar",
    "gu": "guj", "kn": "kan", "ml": "mal", "pa": "pan", "or": "ori", "as": "asm",
    "ne": "nep", "sa": "san", "ur": "urd",
}

# ----------------------------------------------------------------- tokenisation
#
# Indic scripts write vowels and nasalisation as *combining marks*, and Python's ``\w``
# excludes them (they are Unicode Mn/Mc, for which ``str.isalpha()`` is False). A naive
# ``[^\W\d_]+`` therefore splits an Indic word at its own diacritics:
#
#   "ಎಂದರೆ"  ->  ["ಎ", "ದರ"]        (loses the anusvara ಂ and the vowel sign ೆ)
#   "ಏನು"    ->  ["ಏನ"]             (loses ು)
#
# That is not cosmetic. Those two words mean "means" and "what", so the fragments became
# ordinary vocabulary entries and BM25 started ranking passages by their *question frame*:
# a Kannada "what is X" query matched a passage titled "What is ringworm?" ahead of the
# passage that actually answered it. The ZWNJ/ZWJ joiners used in conjuncts (ರಿಂಗ್‌ವರ್ಮ್)
# split words the same way.
#
# So a token is "one letter followed by any run of letters, marks and joiners". The mark
# ranges are derived from Unicode data rather than hand-listed, so they cannot drift out of
# step with SCRIPT_RANGES.
_JOINERS = "\u200c\u200d"


def _mark_class() -> str:
    """Regex character-class body covering combining marks in the corpus scripts."""
    cps = [
        cp
        for _name, lo, hi in SCRIPT_RANGES
        for cp in range(lo, hi + 1)
        if unicodedata.category(chr(cp)) in ("Mn", "Mc")
    ]
    out: list[str] = []
    start = prev = None
    for cp in sorted(cps):
        if start is None:
            start = prev = cp
        elif cp == prev + 1:
            prev = cp
        else:
            out.append(f"\\u{start:04x}" + (f"-\\u{prev:04x}" if prev > start else ""))
            start = prev = cp
    if start is not None:
        out.append(f"\\u{start:04x}" + (f"-\\u{prev:04x}" if prev > start else ""))
    return "".join(out)


_MARKS = _mark_class()
_LETTER = r"[^\W\d_]"
_TOKEN_RE = re.compile(
    rf"{_LETTER}(?:{_LETTER}|[{_MARKS}{_JOINERS}])*|\d+(?:[.,]\d+)?",
    re.UNICODE,
)
_WS_RE = re.compile(r"\s+", re.UNICODE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*(?:\s*%)?", re.UNICODE)

# Sentence terminators across the corpus scripts: Latin, Devanagari danda and double
# danda, Arabic full stop.
TERMINATORS = ".!?\u0964\u0965\u06d4"

# Sentence terminators: Latin, Devanagari danda / double danda, Arabic full stop,
# plus CJK-style stops that occasionally leak through machine translation.
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?\u0964\u0965\u06D4\u3002\uFF01\uFF1F])[\s\u200b]+|\n{2,}",
    re.UNICODE,
)
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "fig", "no", "vol", "inc", "ltd", "co", "u.s", "u.k", "approx",
}

STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if of in on at to for with without from by as is are was were be been being
    do does did done have has had having this that these those it its he she they them his her their
    what which who whom when where why how many much more most other some such no nor not only own
    same so than too very can will just should now about into over after before between during under
    above below out off again further then once here there all any both each few
    """.split()
) | frozenset(
    # High-frequency Hindi / Marathi / Nepali function words (Devanagari)
    """
    का की के को है हैं था थी थे में से पर और या नहीं यह वह जो कि एक भी हो हुआ हुई कर करने किया लिए
    क्या कौन कब कहाँ क्यों कैसे कितना कितने तथा एवं अपने अपनी उस इस
    """.split()
) | frozenset(
    # Bengali / Assamese
    """
    এই সেই একটি এবং বা না নয় হয় হল হলো ছিল আছে করে করা হয়েছে থেকে জন্য সঙ্গে
    কি কী কে কেন কিভাবে কত কোথায় কখন কোন এর তার তাদের কিন্তু বেশি কিছু সব খুব আপনি আমি
    """.split()
) | frozenset(
    # Tamil.  "என்பது" ("is that which") opens most Tamil definition questions, so it has
    # to be a stop word or every such question anchors on its own frame.
    """
    ஒரு இந்த அந்த மற்றும் அல்லது இல்லை என்ன எப்படி எத்தனை ஆக உள்ள உள்ளது ஆகும்
    என்பது என்று என்றால் எது யார் எங்கே ஏன் எப்போது ஆனால் இருந்து அவர் அவள் அது இது
    அதிக சில எல்லா மிக நீங்கள் நான் பற்றி
    """.split()
) | frozenset(
    # Kannada.  "ಎಂದರೇನು" is ಎಂದರೆ+ಏನು fused into one word and is how Kannada asks
    # "what is", so the fused form needs listing alongside the parts.
    """
    ಇದು ಅದು ಈ ಆ ಒಂದು ಮತ್ತು ಅಥವಾ ಇಲ್ಲ ಅಲ್ಲ ಆಗಿದೆ ಇದೆ ಇರುವ ಆಗಿರುವ
    ಎಂದು ಎಂದರೆ ಎಂದರೇನು ಏನೆಂದರೆ ಏನು ಯಾವ ಯಾವುದು ಆದರೆ ಇಂದ ಗೆ ಅವರು ಅವನು ಅವಳು
    ಮಾಡಲು ಹೆಚ್ಚು ಕೆಲವು ಎಲ್ಲಾ ತುಂಬಾ ಸಹ ನೀವು ನಾನು ಬಗ್ಗೆ
    """.split()
) | frozenset(
    # Telugu.  "అంటే ఏమిటి" is the standard "what is" frame.
    """
    ఇది అది ఈ ఆ ఒక మరియు లేదా కాదు లేదు ఉంది ఉన్న అయిన
    అని అంటే ఏమిటి ఏది ఎవరు ఎక్కడ ఎందుకు ఎలా ఎప్పుడు కానీ నుండి కు వారు అతను ఆమె
    చేయడానికి ఎక్కువ కొన్ని అన్ని చాలా కూడా మీరు నేను గురించి
    """.split()
) | frozenset(
    # Malayalam
    """
    ഇത് അത് ഈ ആ ഒരു ഉം അല്ലെങ്കിൽ അല്ല ഇല്ല ആണ് ആകുന്നു ഉണ്ട്
    എന്ന് എന്നാൽ എന്താണ് എന്ത് ഏത് ആര് എവിടെ എന്തുകൊണ്ട് എങ്ങനെ എപ്പോൾ പക്ഷെ
    നിന്ന് അവർ അവൻ അവൾ കൂടുതൽ ചില എല്ലാ വളരെ നിങ്ങൾ ഞാൻ കുറിച്ച്
    """.split()
) | frozenset(
    # Gujarati
    """
    આ તે એક અને અથવા નથી છે હતું હતી હોય માટે થી પર કે નું ની ના
    શું કોણ ક્યાં કેમ કેવી કેટલું ક્યારે તેઓ તેમણે વધુ કેટલાક બધા ખૂબ પણ તમે હું વિશે
    """.split()
) | frozenset(
    # Punjabi (Gurmukhi)
    """
    ਇਹ ਉਹ ਇੱਕ ਅਤੇ ਜਾਂ ਨਹੀਂ ਹੈ ਹਨ ਸੀ ਸਨ ਲਈ ਤੋਂ ਤੇ ਕਿ ਦਾ ਦੀ ਦੇ ਨੂੰ ਵਿੱਚ
    ਕੀ ਕੌਣ ਕਿੱਥੇ ਕਿਉਂ ਕਿਵੇਂ ਕਿੰਨਾ ਕਦੋਂ ਉਹਨਾਂ ਵੱਧ ਕੁਝ ਸਾਰੇ ਬਹੁਤ ਵੀ ਤੁਸੀਂ ਮੈਂ ਬਾਰੇ
    """.split()
) | frozenset(
    # Odia
    """
    ଏହା ସେ ଏକ ଏବଂ କିମ୍ବା ନାହିଁ ଅଛି ଥିଲା ପାଇଁ ରୁ ରେ କି ର ଙ୍କ
    କଣ କିଏ କେଉଁ କାହିଁକି କିପରି କେତେ କେବେ ସେମାନେ ଅଧିକ କିଛି ସବୁ ବହୁତ ମଧ୍ୟ ତୁମେ ମୁଁ ବିଷୟରେ
    """.split()
) | frozenset(
    # Nepali (Devanagari, but a different verb system from Hindi)
    """ छ छन् छैन हो होइन र वा तथा को मा लाई बाट के कसरी कति कहाँ किन कुन """.split()
) | frozenset(
    # Urdu
    """
    کا کی کے کو ہے ہیں تھا تھی میں سے پر اور یا نہیں یہ وہ جو کہ ایک بھی
    کیا کون کہاں کیوں کیسے کتنا کب کچھ سب بہت آپ میں بارے
    """.split()
)

WH_WORDS: frozenset[str] = frozenset(
    """
    what which who whom whose when where why how
    क्या कौन कब कहाँ कहां क्यों कैसे कितना कितने किसका
    কি কে কখন কোথায় কেন কিভাবে কত
    என்ன யார் எப்போது எங்கே ஏன் எப்படி எத்தனை
    ఏమి ఎవరు ఎప్పుడు ఎక్కడ ఎందుకు ఎలా ఎంత
    ಏನು ಯಾರು ಯಾವಾಗ ಎಲ್ಲಿ ಏಕೆ ಹೇಗೆ ಎಷ್ಟು
    എന്ത് ആര് എപ്പോൾ എവിടെ എന്തുകൊണ്ട് എങ്ങനെ എത്ര
    کیا کون کب کہاں کیوں کیسے کتنا
    """.split()
)

# Cue lexicons for the query router.
#
# Single-word cues and multi-word cues are kept in SEPARATE structures on purpose.
# Storing "how many" in a whitespace-split token set silently adds bare "how" as a cue,
# which classifies every "how does X work" question as NUMERIC. The same mistake with
# "what is" adds bare "is", which matches almost every English sentence ever written.
# Token sets are matched against tokens; phrase tuples are matched against the raw
# lowercased query.

DEFINITION_CUES: frozenset[str] = frozenset(
    """
    define definition definitions acronym abbreviation
    अर्थ मतलब परिभाषा
    অর্থ মানে সংজ্ঞা
    அர்த்தம் பொருள் வரையறை என்பது
    ಎಂದರೇನು ಎಂದರೆ ಅರ್ಥ ವ್ಯಾಖ್ಯಾನ
    అంటే అర్థం నిర్వచనం
    എന്താണ് അർത്ഥം നിർവചനം
    અર્થ વ્યાખ્યા
    ਅਰਥ ਮਤਲਬ ਪਰਿਭਾਸ਼ਾ
    ଅର୍ଥ ପରିଭାଷା
    """.split()
)

DEFINITION_PHRASES: tuple[str, ...] = (
    "what is", "what are", "what's", "whats", "what does", "meaning of",
    "definition of", "stands for", "known as", "referred to as",
    "क्या है", "क्या होता है", "কি", "কী", "என்றால் என்ன",
)
# Indic languages put the interrogative at the *end* ("ಕಾರ್ಪೊರೇಷನ್ ಎಂದರೇನು?"), so they are
# matched as token cues above rather than as prefixes here.

NUMERIC_CUES: frozenset[str] = frozenset(
    """
    cost price prices average percent percentage calories kilograms grams
    temperature weight length distance duration salary
    कितना कितने कीमत औसत प्रतिशत
    কত দাম গড় শতাংশ
    எத்தனை விலை சராசரி சதவீதம்
    ఎంత ధర సగటు
    ಎಷ್ಟು ಬೆಲೆ
    എത്ര വില
    کتنا قیمت اوسط
    """.split()
)

NUMERIC_PHRASES: tuple[str, ...] = (
    "how many", "how much", "how long", "how far", "how old", "how tall", "how deep",
    "what percentage", "what percent", "average number", "number of", "total number",
)

LIST_CUES: frozenset[str] = frozenset(
    "list types kinds varieties examples steps stages symptoms causes benefits uses ways".split()
)

LIST_PHRASES: tuple[str, ...] = (
    "types of", "kinds of", "list of", "examples of", "steps to", "ways to",
    "different types", "what are the",
)

YESNO_CUES: frozenset[str] = frozenset(
    "is are was were do does did can could should would will has have had".split()
)


def normalize(text: str, *, casefold: bool = True) -> str:
    """NFKC normalise, collapse whitespace, strip zero-width joiners."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _WS_RE.sub(" ", text).strip()
    return text.casefold() if casefold else text


# A Latin run this long inside Indic text is a loanword or a proper noun, not a change of
# language. Two letters of Kannada, on the other hand, cannot appear inside an English
# question by accident.
_MIN_INDIC_RUN = 2


def detect_script(text: str) -> str:
    """Dominant script, with Latin treated as the weaker claim.

    Counting codepoints and taking the maximum reads "Cooperation ಎಂದರೆ ಏನು?" as Latin,
    because the loanword is longer than the two Kannada words around it - and then the
    whole turn answers in English. Indic text embeds Latin words constantly (brand names,
    technical terms, anything untranslated); English text does not embed Kannada. So any
    substantial non-Latin script present wins, and Latin is the fallback.

    Combining marks count toward their script here. ``str.isalpha()`` is False for them, so
    skipping non-alphabetic characters was silently discounting Indic text by roughly the
    number of vowels it contains.
    """
    counts: dict[str, int] = {}
    latin = 0
    for ch in text:
        cp = ord(ch)
        hit = False
        for name, lo, hi in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                hit = True
                break
        if not hit and ch.isalpha() and cp < 0x0250:
            latin += 1

    if counts:
        name, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= _MIN_INDIC_RUN:
            return name
    if latin:
        return "latin"
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "latin"


def normalize_lang(code: str | None) -> str | None:
    """An ISO-639-1 code from anything callers might supply (``hi``, ``hin``, ``hi-IN``)."""
    if not code:
        return None
    h = code.lower().replace("_", "-").split("-")[0]
    if h in LANG_NAMES:
        return h
    for iso1, iso3 in ISO1_TO_ISO3.items():
        if h == iso3:
            return iso1
    return None


def detect_lang(text: str, hint: str | None = None) -> str:
    """Best-effort language code. An explicit hint always wins."""
    return normalize_lang(hint) or SCRIPT_DEFAULT_LANG.get(detect_script(text), "en")


# ISO-639-1 -> the script that language is written in.
LANG_SCRIPT: dict[str, str] = {
    "en": "latin", "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
    "sa": "devanagari", "bn": "bengali", "as": "bengali", "pa": "gurmukhi",
    "gu": "gujarati", "or": "odia", "ta": "tamil", "te": "telugu", "kn": "kannada",
    "ml": "malayalam", "ur": "arabic",
}


def resolve_lang(text: str, claim: str | None) -> str:
    """Reconcile a *claimed* language with the script the text is actually written in.

    Used for speech-to-text, where the claim comes from the recogniser rather than from the
    user. A claim is accepted only when it is consistent with the script in front of us:
    Scribe reporting English while returning Kannada characters is a misdetection, and
    honouring it means retrieving and answering in the wrong language.

    Claims within the same script are trusted, because they carry information the script
    cannot: Hindi, Marathi, Nepali and Sanskrit all look like Devanagari, so a claim of
    Marathi is strictly better than the script's default guess of Hindi.
    """
    code = normalize_lang(claim)
    script = detect_script(text)
    if code and LANG_SCRIPT.get(code) == script:
        return code
    return SCRIPT_DEFAULT_LANG.get(script, code or "en")


def tokenize(text: str) -> list[str]:
    """Unicode-aware word + number tokens, casefolded.

    Joiners hold a token together during matching but are then dropped, so that a word
    written with an explicit ZWNJ and the same word written without it are one vocabulary
    entry rather than two.
    """
    out = []
    for tok in _TOKEN_RE.findall(text.casefold()):
        t = tok.strip(_JOINERS).replace("\u200c", "").replace("\u200d", "")
        if t:
            out.append(t)
    return out


def content_tokens(text: str) -> list[str]:
    """Tokens that carry the *subject* of a query, with function words removed.

    Interrogatives are excluded as well as stop words. They are the frame of the question,
    never its content, and callers use this to decide whether evidence is on-topic - so a
    shared "what"/"ಏನು" must not read as agreement about the subject. Keeping them in is
    how a Kannada "what is X" query came back with a passage titled "What is ringworm?"
    and a topicality verdict claiming two lexical anchors.
    """
    return [
        t
        for t in tokenize(text)
        if t not in STOPWORDS and t not in WH_WORDS and len(t) > 1
    ]


def numbers_in(text: str) -> list[str]:
    """Normalised numeric literals (used by the hallucination check)."""
    out = []
    for m in _NUM_RE.findall(text):
        cleaned = m.replace(",", "").replace(" ", "").rstrip("%")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        out.append(f"{val:g}")
    return out


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    start: int
    end: int

    @property
    def n_chars(self) -> int:
        return self.end - self.start


def split_sentences(text: str, *, min_chars: int = 2) -> list[Sentence]:
    """Sentence split with character offsets preserved.

    Offsets matter: the chunk lattice records ``char_start``/``char_end`` for
    every chunk so the UI can highlight the exact span inside the source passage
    and so parent expansion is a pure slice rather than a re-search.
    """
    if not text:
        return []
    pieces: list[Sentence] = []
    cursor = 0
    for raw in _SENT_SPLIT_RE.split(text):
        if raw is None:
            continue
        idx = text.find(raw, cursor) if raw else cursor
        if idx < 0:
            idx = cursor
        stripped = raw.strip()
        if stripped:
            offset = idx + (len(raw) - len(raw.lstrip()))
            pieces.append(Sentence(stripped, offset, offset + len(stripped)))
        cursor = idx + len(raw)

    # Re-attach fragments produced by abbreviations ("Dr. Rao" -> one sentence).
    merged: list[Sentence] = []
    for s in pieces:
        if merged:
            prev = merged[-1]
            tail = prev.text.rstrip(".").split()[-1].casefold() if prev.text.rstrip(".").split() else ""
            if (tail in _ABBREV and prev.text.endswith(".")) or len(s.text) < min_chars:
                merged[-1] = Sentence(f"{prev.text} {s.text}".strip(), prev.start, s.end)
                continue
        merged.append(s)
    return merged or [Sentence(text.strip(), 0, len(text))]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(needle: set[str], haystack: set[str]) -> float:
    """Fraction of ``needle`` present in ``haystack`` - asymmetric on purpose.

    Groundedness asks "is every token of the claim supported by the evidence",
    which is containment, not Jaccard.
    """
    if not needle:
        return 1.0
    return len(needle & haystack) / len(needle)


def truncate(text: str, limit: int, suffix: str = "\u2026") -> str:
    """Trim to ``limit`` characters, never mid-word.

    Prefers a sentence boundary, then a word boundary, and only cuts mid-token when the
    text contains no break at all. This matters more than it looks: Indic scripts pack
    more meaning per character, so a naive character cut regularly lands inside a word and
    produces something that reads as corrupted rather than abbreviated.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    for term in TERMINATORS:
        idx = window.rfind(term)
        if idx > limit * 0.5:
            return window[: idx + 1]
    sp = window.rfind(" ")
    if sp > limit * 0.35:
        window = window[:sp]
    return window.rstrip(" ,;:.\u0964\u0965") + suffix


__all__ = [
    "DEFINITION_CUES",
    "DEFINITION_PHRASES",
    "ISO1_TO_ISO3",
    "LANG_NAMES",
    "LIST_CUES",
    "LIST_PHRASES",
    "NUMERIC_CUES",
    "NUMERIC_PHRASES",
    "LANG_SCRIPT",
    "STOPWORDS",
    "TERMINATORS",
    "Sentence",
    "WH_WORDS",
    "XI_LANGS",
    "YESNO_CUES",
    "containment",
    "content_tokens",
    "detect_lang",
    "detect_script",
    "jaccard",
    "normalize",
    "normalize_lang",
    "numbers_in",
    "resolve_lang",
    "split_sentences",
    "tokenize",
    "truncate",
]
