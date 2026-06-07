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

async function mount(): Promise<void> {
  const gpEl = document.getElementById("gp");
  const gaussianEl = document.getElementById("gaussian");
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
}

setupTabs();
void mount();
