/**
 * The 15 languages SETU speaks.
 *
 * `native` is what the language calls itself - a Kannada speaker looking for their
 * language scans for "ಕನ್ನಡ", not for "Kannada", so the picker shows both.
 * `bcp47` drives browser speech recognition and speech synthesis voice selection.
 * `example` is a real question in that language, used to seed the input when someone
 * picks a language they cannot currently type in.
 */
export interface Lang {
  code: string;
  name: string;
  native: string;
  bcp47: string;
  example: string;
}

export const LANGUAGES: Lang[] = [
  { code: "en", name: "English", native: "English", bcp47: "en-IN", example: "what is a corporation" },
  { code: "hi", name: "Hindi", native: "हिन्दी", bcp47: "hi-IN", example: "कॉर्पोरेशन क्या है?" },
  { code: "bn", name: "Bengali", native: "বাংলা", bcp47: "bn-IN", example: "কর্পোরেশন কী?" },
  { code: "ta", name: "Tamil", native: "தமிழ்", bcp47: "ta-IN", example: "நிறுவனம் என்றால் என்ன?" },
  { code: "te", name: "Telugu", native: "తెలుగు", bcp47: "te-IN", example: "కార్పొరేషన్ అంటే ఏమిటి?" },
  { code: "kn", name: "Kannada", native: "ಕನ್ನಡ", bcp47: "kn-IN", example: "ಕಾರ್ಪೊರೇಶನ್ ಎಂದರೇನು?" },
  { code: "ml", name: "Malayalam", native: "മലയാളം", bcp47: "ml-IN", example: "കോർപ്പറേഷൻ എന്താണ്?" },
  { code: "mr", name: "Marathi", native: "मराठी", bcp47: "mr-IN", example: "कॉर्पोरेशन म्हणजे काय?" },
  { code: "gu", name: "Gujarati", native: "ગુજરાતી", bcp47: "gu-IN", example: "કોર્પોરેશન શું છે?" },
  { code: "pa", name: "Punjabi", native: "ਪੰਜਾਬੀ", bcp47: "pa-IN", example: "ਕਾਰਪੋਰੇਸ਼ਨ ਕੀ ਹੈ?" },
  { code: "or", name: "Odia", native: "ଓଡ଼ିଆ", bcp47: "or-IN", example: "କର୍ପୋରେଶନ କଣ?" },
  { code: "as", name: "Assamese", native: "অসমীয়া", bcp47: "as-IN", example: "কৰ্পʼৰেশন কি?" },
  { code: "ne", name: "Nepali", native: "नेपाली", bcp47: "ne-NP", example: "कर्पोरेशन के हो?" },
  { code: "sa", name: "Sanskrit", native: "संस्कृतम्", bcp47: "hi-IN", example: "निगमः किम्?" },
  { code: "ur", name: "Urdu", native: "اردو", bcp47: "ur-IN", example: "کارپوریشن کیا ہے؟" },
];

export const LANG_BY_CODE: Record<string, Lang> = Object.fromEntries(
  LANGUAGES.map((l) => [l.code, l]),
);

export function langName(code: string | null | undefined): string {
  if (!code) return "auto";
  return LANG_BY_CODE[code]?.name ?? code;
}

export function langNative(code: string | null | undefined): string {
  if (!code) return "Auto-detect";
  return LANG_BY_CODE[code]?.native ?? code;
}

export function bcp47(code: string | null | undefined): string {
  if (!code) return "en-IN";
  return LANG_BY_CODE[code]?.bcp47 ?? "en-IN";
}
