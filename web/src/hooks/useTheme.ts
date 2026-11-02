import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "setu.theme";

/**
 * Light by default, dark on request, remembered.
 *
 * Light is the default deliberately: this is a public question box, and a warm paper-white
 * page reads as calm and approachable where a black instrument panel reads as an internal
 * developer tool. An explicit choice always wins over the OS preference, because someone who
 * flips the switch here means it for this site.
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem(KEY);
    return saved === "dark" || saved === "light" ? saved : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(KEY, theme);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#08090d" : "#f6f5f2");
  }, [theme]);

  const toggle = useCallback(() => setTheme((t) => (t === "light" ? "dark" : "light")), []);
  return { theme, toggle, setTheme };
}
