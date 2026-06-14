/**
 * Entry point: tab switching + lazy demo mounting.
 *
 * Each demo is imported lazily so a load failure in one tab does not blank the
 * other, and so the first paint (the tab shell) is instant.
 */

function setupTabs(): void {
  const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>(".tab"));
  const panels = Array.from(document.querySelectorAll<HTMLElement>(".panel"));
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      for (const t of tabs) t.classList.toggle("active", t === tab);
      for (const p of panels) p.classList.toggle("active", p.id === name);
    });
  }
}

function setupFullscreenToggle(): void {
  const btn = document.querySelector<HTMLButtonElement>(".fullscreen-toggle");
  if (!btn) return;

  const canFullscreen =
    typeof document.documentElement.requestFullscreen === "function" &&
    typeof document.exitFullscreen === "function";
  if (!canFullscreen) {
    btn.disabled = true;
    btn.textContent = "Fullscreen unavailable";
    return;
  }

  const update = () => {
    const active = Boolean(document.fullscreenElement);
    btn.textContent = active ? "Exit full screen" : "Full screen";
    btn.setAttribute("aria-pressed", String(active));
  };

  btn.addEventListener("click", () => {
    void (async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await document.documentElement.requestFullscreen();
      } catch {
        update();
      }
    })();
  });
  document.addEventListener("fullscreenchange", update);
  update();
}

function setupAceInfoModal(): void {
  const openBtn = document.querySelector<HTMLButtonElement>(".ace-info-toggle");
  const modal = document.getElementById("ace-modal");
  const closeBtn = document.querySelector<HTMLButtonElement>(".ace-modal-close");
  if (!openBtn || !modal || !closeBtn) return;

  const close = () => {
    modal.hidden = true;
    openBtn.setAttribute("aria-expanded", "false");
    openBtn.focus();
  };

  const open = () => {
    modal.hidden = false;
    openBtn.setAttribute("aria-expanded", "true");
    closeBtn.focus();
  };

  openBtn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) close();
  });
}

function setupReferencesModal(): void {
  const openBtn = document.querySelector<HTMLButtonElement>(".footer-refs-btn");
  const modal = document.getElementById("refs-modal");
  if (!openBtn || !modal) return;
  // Scope the close button to this modal — the per-tab explainers also use
  // `.ace-modal-close`, so a document-wide query would be order-dependent.
  const closeBtn = modal.querySelector<HTMLButtonElement>(".ace-modal-close");
  if (!closeBtn) return;

  const close = () => {
    modal.hidden = true;
    openBtn.setAttribute("aria-expanded", "false");
    openBtn.focus();
  };
  const open = () => {
    modal.hidden = false;
    openBtn.setAttribute("aria-expanded", "true");
    closeBtn.focus();
  };
  openBtn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) close();
  });
}

async function mount(): Promise<void> {
  const gpEl = document.getElementById("gp");
  const gaussianEl = document.getElementById("gaussian");
  const sirEl = document.getElementById("sir");
  const boEl = document.getElementById("bo");
  const arbufEl = document.getElementById("arbuf");
  const alineEl = document.getElementById("aline");
  if (gpEl) {
    try {
      const { mountGP } = await import("./gp/demo");
      await mountGP(gpEl);
    } catch (err) {
      gpEl.innerHTML = `<p class="loading">Failed to load GP demo: ${String(err)}</p>`;
    }
  }
  if (gaussianEl) {
    try {
      const { mountGaussian } = await import("./gaussian/demo");
      await mountGaussian(gaussianEl);
    } catch (err) {
      gaussianEl.innerHTML = `<p class="loading">Failed to load Gaussian demo: ${String(err)}</p>`;
    }
  }
  if (sirEl) {
    try {
      const { mountSIR } = await import("./sir/demo");
      await mountSIR(sirEl);
    } catch (err) {
      sirEl.innerHTML = `<p class="loading">Failed to load SIR demo: ${String(err)}</p>`;
    }
  }
  if (boEl) {
    try {
      const { mountBO } = await import("./bo/demo");
      await mountBO(boEl);
    } catch (err) {
      boEl.innerHTML = `<p class="loading">Failed to load BO demo: ${String(err)}</p>`;
    }
  }
  if (arbufEl) {
    try {
      const { mountArbuf } = await import("./arbuf/demo");
      await mountArbuf(arbufEl);
    } catch (err) {
      arbufEl.innerHTML = `<p class="loading">Failed to load AR-buffer demo: ${String(err)}</p>`;
    }
  }
  if (alineEl) {
    try {
      const { mountAline } = await import("./aline/demo");
      await mountAline(alineEl);
    } catch (err) {
      alineEl.innerHTML = `<p class="loading">Failed to load ALINE demo: ${String(err)}</p>`;
    }
  }
}

setupTabs();
setupFullscreenToggle();
setupAceInfoModal();
setupReferencesModal();
void mount();
