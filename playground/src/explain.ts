/**
 * Per-tab "?" explainer: a small inline button at the end of each tab's hint
 * line, opening a modal that explains the tab's task, what ACE is doing, and
 * how it compares to the classical approach. Reuses the global `ace-modal`
 * chrome from index.html so the dialogs match the "What is ACE?" one; only the
 * button and section styles are added here.
 */

export interface ExplainSpec {
  title: string;
  html: string; // <h3>/<p> sections; end with aceFooter(...)
}

/** Shared attribution footer; `extra` adds a tab-specific citation sentence. */
export function aceFooter(extra = ""): string {
  return (
    '<p class="explain-footer">Based on Chang et al. (2025), ' +
    "<em>Amortized Probabilistic Conditioning for Optimization, Simulation and Inference</em> " +
    '(AISTATS 2025) — <a href="https://acerbilab.github.io/amortized-conditioning-engine/">project page</a>.' +
    (extra ? ` ${extra}` : "") +
    "</p>"
  );
}

const CSS = `
.info-btn { display: inline-flex; align-items: center; justify-content: center;
  width: 18px; height: 18px; padding: 0; margin-left: 8px; vertical-align: text-bottom;
  border: 1px solid var(--line); border-radius: 50%; background: #fff;
  color: var(--muted); font: 600 12px/1 system-ui; cursor: pointer; }
.info-btn:hover { color: var(--accent); border-color: var(--accent); }
.explain-modal h3 { margin: 14px 0 0; font-size: 13.5px; }
.explain-modal h3:first-child { margin-top: 2px; }
.explain-modal p { margin: 6px 0 0; color: #374151; }
`;
// .explain-footer styling lives in index.html next to the shared ace-modal chrome
// (the global "What is ACE?" modal uses the same footer class).

function injectCss(): void {
  if (document.getElementById("explain-style")) return;
  const s = document.createElement("style");
  s.id = "explain-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

/** Append a "?" button to `anchor` (a tab's hint line) and wire up its modal. */
export function addInfoButton(anchor: HTMLElement, spec: ExplainSpec): void {
  injectCss();

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "info-btn";
  btn.textContent = "?";
  btn.title = spec.title;
  btn.setAttribute("aria-haspopup", "dialog");
  btn.setAttribute("aria-expanded", "false");
  anchor.appendChild(btn);

  const modal = document.createElement("div");
  modal.className = "ace-modal explain-modal";
  modal.hidden = true;
  modal.innerHTML = `
    <div class="ace-modal-card" role="dialog" aria-modal="true">
      <div class="ace-modal-head">
        <h2></h2>
        <button class="ace-modal-close" type="button">Close</button>
      </div>
      <div class="explain-body"></div>
    </div>`;
  modal.querySelector("h2")!.textContent = spec.title;
  modal.querySelector<HTMLDivElement>(".explain-body")!.innerHTML = spec.html;
  // Sibling of the hint inside the tab root, so it mounts/unmounts with the tab.
  (anchor.parentElement ?? document.body).appendChild(modal);

  const closeBtn = modal.querySelector<HTMLButtonElement>(".ace-modal-close")!;
  const close = () => {
    modal.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    btn.focus();
  };
  const open = () => {
    modal.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    closeBtn.focus();
  };
  btn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) close();
  });
}
