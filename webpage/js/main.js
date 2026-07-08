/* Page glue: KaTeX auto-render, TOC scroll-spy, copy-citation. */
"use strict";

/* headless-debug error overlay: open the page with #debug to surface any runtime error visually */
if (location.hash.includes("debug")) {
  window.addEventListener("error", (e) => {
    let box = document.getElementById("_errbox");
    if (!box) {
      box = document.createElement("div");
      box.id = "_errbox";
      box.style.cssText =
        "position:fixed;top:0;left:0;right:0;z-index:9999;background:#7f1d1d;color:#fff;" +
        "font:12px monospace;padding:8px 12px;white-space:pre-wrap";
      document.documentElement.appendChild(box);
    }
    box.textContent += "ERROR: " + e.message + " @ " + (e.filename || "?") + ":" + (e.lineno || "?") + "\n";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  /* math */
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }

  /* TOC scroll-spy */
  const tocLinks = Array.from(document.querySelectorAll(".toc a"));
  const targets = tocLinks
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);
  function spy() {
    let active = 0;
    const y = window.scrollY + 120;
    targets.forEach((t, i) => { if (t.offsetTop <= y) active = i; });
    tocLinks.forEach((a, i) => a.classList.toggle("active", i === active));
  }
  window.addEventListener("scroll", spy, { passive: true });
  spy();

  /* copy citation */
  const btn = document.getElementById("btn-copy-cite");
  if (btn) {
    btn.addEventListener("click", () => {
      const txt = document.getElementById("bibtex").textContent.replace(/copy\s*$/, "").trim();
      navigator.clipboard.writeText(txt).then(() => {
        btn.textContent = "copied ✓";
        setTimeout(() => (btn.textContent = "copy"), 1500);
      });
    });
  }
});
