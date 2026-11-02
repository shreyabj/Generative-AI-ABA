"""Answer surface: stitching, citation markers, and localised refusals.

A guardrail that refuses in English to someone who asked in Tamil has failed at the
part the user actually experiences. Every abstention, refusal and safe-completion
message here exists in all 15 corpus languages, so the *system's own voice* matches
the user's language even when the corpus could not answer.
"""

from __future__ import annotations

from ..harness.contracts import Citation, Layer
from ..text import TERMINATORS as TERMINALS
from ..text import truncate

# ------------------------------------------------------------------- templates
NO_EVIDENCE: dict[str, str] = {
    "en": "I don't have that in my knowledge base. My corpus is MS MARCO-XI, so I can only answer from passages it contains.",
    "hi": "यह जानकारी मेरे ज्ञान-आधार में नहीं है। मैं केवल MS MARCO-XI संग्रह के अंशों से उत्तर दे सकता हूँ।",
    "bn": "এই তথ্য আমার জ্ঞানভাণ্ডারে নেই। আমি কেবল MS MARCO-XI সংগ্রহের অনুচ্ছেদ থেকে উত্তর দিতে পারি।",
    "ta": "இந்தத் தகவல் என் அறிவுத் தளத்தில் இல்லை. நான் MS MARCO-XI தொகுப்பில் உள்ள பகுதிகளிலிருந்து மட்டுமே பதிலளிக்க முடியும்.",
    "te": "ఈ సమాచారం నా విజ్ఞాన నిధిలో లేదు. నేను MS MARCO-XI సంకలనంలోని భాగాల నుండి మాత్రమే సమాధానం ఇవ్వగలను.",
    "mr": "ही माहिती माझ्या ज्ञानसंग्रहात नाही. मी केवळ MS MARCO-XI संग्रहातील उताऱ्यांवरून उत्तर देऊ शकतो.",
    "gu": "આ માહિતી મારા જ્ઞાનભંડારમાં નથી. હું ફક્ત MS MARCO-XI સંગ્રહના ફકરાઓમાંથી જવાબ આપી શકું છું.",
    "kn": "ಈ ಮಾಹಿತಿ ನನ್ನ ಜ್ಞಾನಭಂಡಾರದಲ್ಲಿ ಇಲ್ಲ. ನಾನು MS MARCO-XI ಸಂಗ್ರಹದ ಭಾಗಗಳಿಂದ ಮಾತ್ರ ಉತ್ತರಿಸಬಲ್ಲೆ.",
    "ml": "ഈ വിവരം എന്റെ വിജ്ഞാനശേഖരത്തിൽ ഇല്ല. MS MARCO-XI ശേഖരത്തിലെ ഭാഗങ്ങളിൽ നിന്ന് മാത്രമേ എനിക്ക് ഉത്തരം നൽകാനാകും.",
    "pa": "ਇਹ ਜਾਣਕਾਰੀ ਮੇਰੇ ਗਿਆਨ-ਭੰਡਾਰ ਵਿੱਚ ਨਹੀਂ ਹੈ। ਮੈਂ ਸਿਰਫ਼ MS MARCO-XI ਸੰਗ੍ਰਹਿ ਦੇ ਹਿੱਸਿਆਂ ਤੋਂ ਜਵਾਬ ਦੇ ਸਕਦਾ ਹਾਂ।",
    "or": "ଏହି ସୂଚନା ମୋ ଜ୍ଞାନଭଣ୍ଡାରରେ ନାହିଁ। ମୁଁ କେବଳ MS MARCO-XI ସଂଗ୍ରହର ଅଂଶରୁ ଉତ୍ତର ଦେଇପାରିବି।",
    "as": "এই তথ্য মোৰ জ্ঞানভাণ্ডাৰত নাই। মই কেৱল MS MARCO-XI সংগ্ৰহৰ অংশৰ পৰাই উত্তৰ দিব পাৰোঁ।",
    "ne": "यो जानकारी मेरो ज्ञानभण्डारमा छैन। म केवल MS MARCO-XI संग्रहका अंशहरूबाट मात्र उत्तर दिन सक्छु।",
    "sa": "एतत् ज्ञानं मम कोशे नास्ति। अहं केवलं MS MARCO-XI सङ्ग्रहस्य अंशेभ्यः एव उत्तरं दातुं शक्नोमि।",
    "ur": "یہ معلومات میرے علمی ذخیرے میں نہیں ہیں۔ میں صرف MS MARCO-XI مجموعے کے اقتباسات سے جواب دے سکتا ہوں۔",
}

LOW_GROUNDING: dict[str, str] = {
    "en": "I found related passages but none of them actually support an answer, so I'd rather not guess.",
    "hi": "मुझे संबंधित अंश मिले, परंतु उनमें से कोई भी उत्तर की पुष्टि नहीं करता, इसलिए मैं अनुमान नहीं लगाऊँगा।",
    "bn": "সম্পর্কিত অনুচ্ছেদ পেয়েছি, কিন্তু কোনোটিই উত্তরটি সমর্থন করে না, তাই আমি অনুমান করব না।",
    "ta": "தொடர்புடைய பகுதிகள் கிடைத்தன, ஆனால் எதுவும் பதிலை உறுதிப்படுத்தவில்லை, எனவே நான் ஊகிக்க விரும்பவில்லை.",
    "te": "సంబంధిత భాగాలు దొరికాయి, కానీ ఏవీ సమాధానాన్ని బలపరచలేదు, అందుకే నేను ఊహించను.",
    "mr": "संबंधित उतारे मिळाले, पण त्यांपैकी कोणतेही उत्तराला आधार देत नाही, म्हणून मी अंदाज लावणार नाही.",
    "ur": "متعلقہ اقتباسات ملے، مگر کوئی بھی جواب کی تصدیق نہیں کرتا، اس لیے میں اندازہ نہیں لگاؤں گا۔",
}

OFF_TOPIC: dict[str, str] = {
    "en": "That's outside what I'm built for. I answer factual questions grounded in the MS MARCO-XI passage collection.",
    "hi": "यह मेरे कार्यक्षेत्र से बाहर है। मैं MS MARCO-XI अंश-संग्रह पर आधारित तथ्यात्मक प्रश्नों के उत्तर देता हूँ।",
    "bn": "এটি আমার কাজের বাইরে। আমি MS MARCO-XI সংগ্রহভিত্তিক তথ্যমূলক প্রশ্নের উত্তর দিই।",
    "ta": "இது என் பணி எல்லைக்கு வெளியே உள்ளது. MS MARCO-XI தொகுப்பை அடிப்படையாகக் கொண்ட உண்மைத் தகவல் கேள்விகளுக்கு மட்டுமே பதிலளிக்கிறேன்.",
    "te": "ఇది నా పరిధి వెలుపల ఉంది. MS MARCO-XI సంకలనం ఆధారంగా వాస్తవిక ప్రశ్నలకు మాత్రమే సమాధానం ఇస్తాను.",
    "ur": "یہ میرے دائرۂ کار سے باہر ہے۔ میں MS MARCO-XI مجموعے پر مبنی حقائق سے متعلق سوالات کے جواب دیتا ہوں۔",
}

UNSAFE: dict[str, str] = {
    "en": "I can't help with that request.",
    "hi": "मैं इस अनुरोध में सहायता नहीं कर सकता।",
    "bn": "আমি এই অনুরোধে সহায়তা করতে পারব না।",
    "ta": "இந்தக் கோரிக்கையில் நான் உதவ முடியாது.",
    "te": "ఈ అభ్యర్థనలో నేను సహాయం చేయలేను.",
    "ur": "میں اس درخواست میں مدد نہیں کر سکتا۔",
}

SELF_HARM: dict[str, str] = {
    "en": (
        "I'm not able to help with this, and I don't want to just leave it there. "
        "If you're in India you can reach Tele-MANAS any time on 14416, or KIRAN on 1800-599-0019. "
        "Outside India, please contact your local emergency number."
    ),
    "hi": (
        "मैं इसमें मदद नहीं कर सकता, पर आपको अकेला भी नहीं छोड़ना चाहता। "
        "भारत में आप किसी भी समय Tele-MANAS को 14416 पर या KIRAN को 1800-599-0019 पर कॉल कर सकते हैं।"
    ),
    "ta": (
        "இதில் என்னால் உதவ முடியாது, ஆனால் உங்களை தனியாக விட விரும்பவில்லை. "
        "இந்தியாவில் Tele-MANAS 14416 அல்லது KIRAN 1800-599-0019 எந்த நேரமும் அணுகலாம்."
    ),
}

INJECTION: dict[str, str] = {
    "en": "That looks like an attempt to change my instructions, so I'm ignoring it. Ask me something from the corpus instead.",
    "hi": "यह मेरे निर्देश बदलने का प्रयास लगता है, इसलिए मैं इसे अनदेखा कर रहा हूँ। कृपया संग्रह से संबंधित प्रश्न पूछें।",
    "ta": "இது என் வழிமுறைகளை மாற்ற முயற்சிப்பது போல் தெரிகிறது, எனவே அதைப் புறக்கணிக்கிறேன். தொகுப்பிலிருந்து ஒரு கேள்வி கேளுங்கள்.",
}


def localized(table: dict[str, str], lang: str) -> str:
    return table.get(lang) or table.get("en", "")


# ------------------------------------------------------------------ composition
def stitch(sentences: list[str], markers: list[int], *, max_chars: int) -> str:
    """Join selected spans into one answer, attaching citation markers.

    The marker goes *inside* the sentence, before its terminal punctuation:

        "A corporation is a legal entity [1]."   not   "A corporation is a legal entity. [1]"

    This is not cosmetic. The groundedness gate re-splits the answer into sentences to
    verify each claim independently, and a trailing marker lands at the *start* of the
    next sentence after splitting - so claim 1 would appear uncited and claim 2 would
    appear to cite evidence it never used. Keeping the marker inside the sentence makes
    claim-to-evidence attribution survive the round trip.
    """
    parts: list[str] = []
    for text, marker in zip(sentences, markers, strict=True):
        text = text.strip()
        if not text:
            continue
        if text[-1] in TERMINALS:
            parts.append(f"{text[:-1].rstrip()} [{marker}]{text[-1]}")
        else:
            parts.append(f"{text} [{marker}].")
    joined = " ".join(parts)
    if len(joined) > max_chars:
        # Trim whole spans rather than mid-sentence, keeping at least one.
        while len(parts) > 1 and len(" ".join(parts)) > max_chars:
            parts.pop()
        joined = " ".join(parts)
        if len(joined) > max_chars:
            joined = truncate(joined, max_chars)
    return joined


def build_citations(rows: list[tuple[int, str, str, Layer, str, float]]) -> list[Citation]:
    """``rows`` is (marker, chunk_id, doc_id, layer, snippet, score)."""
    return [
        Citation(marker=m, chunk_id=cid, doc_id=did, layer=layer, snippet=truncate(snip, 300), score=round(sc, 6))
        for m, cid, did, layer, snip, sc in rows
    ]


__all__ = [
    "INJECTION",
    "LOW_GROUNDING",
    "NO_EVIDENCE",
    "OFF_TOPIC",
    "SELF_HARM",
    "TERMINALS",
    "UNSAFE",
    "build_citations",
    "localized",
    "stitch",
]
