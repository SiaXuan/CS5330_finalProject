/* ── State ───────────────────────────────────────────────────── */
const PAGE_SIZE = 12;
const state = {
  referenceAsin: null,
  searchHistory: [],
  allResults:    [],
  displayedCount: 0,
};

/* ── DOM refs ────────────────────────────────────────────────── */
const searchScreen  = document.getElementById("search-screen");
const resultsScreen = document.getElementById("results-screen");
const imageGrid     = document.getElementById("image-grid");
const historyList   = document.getElementById("history-list");
const resultsLabel  = document.getElementById("results-label");
const loading       = document.getElementById("loading");
const modalOverlay  = document.getElementById("modal-overlay");
const modalImg      = document.getElementById("modal-img");
const refineText    = document.getElementById("refine-text");
const refineCat     = document.getElementById("refine-cat");

/* ── Helpers ─────────────────────────────────────────────────── */
function showLoading()  { loading.classList.remove("hidden"); }
function hideLoading()  { loading.classList.add("hidden"); }
function showModal()    { modalOverlay.classList.remove("hidden"); refineText.focus(); }
function hideModal()    { modalOverlay.classList.add("hidden"); refineText.value = ""; refineCat.value = ""; }

function showResults() {
  searchScreen.classList.remove("active");
  resultsScreen.classList.add("active");
}

function makeCard(item) {
  const card = document.createElement("div");
  card.className = "img-card";
  card.dataset.asin = item.asin;
  card.dataset.category = item.category;

  const img = document.createElement("img");
  img.src = `/image/${item.asin}`;
  img.alt = item.asin;
  img.loading = "lazy";
  img.onerror = () => {
    img.replaceWith(Object.assign(document.createElement("div"), {
      className: "img-placeholder",
      textContent: "No image",
    }));
  };

  const badge = document.createElement("span");
  badge.className = "cat-badge";
  badge.textContent = item.category;

  const overlay = document.createElement("div");
  overlay.className = "card-overlay";
  overlay.innerHTML = "<span>Refine →</span>";

  card.appendChild(img);
  card.appendChild(badge);
  card.appendChild(overlay);
  card.addEventListener("click", () => openRefineModal(item.asin));
  return card;
}

function appendNextPage() {
  const start = state.displayedCount;
  const end   = Math.min(start + PAGE_SIZE, state.allResults.length);
  if (start >= state.allResults.length) return;

  const frag = document.createDocumentFragment();
  for (let i = start; i < end; i++) {
    frag.appendChild(makeCard(state.allResults[i]));
  }
  // Insert before sentinel
  const sentinel = document.getElementById("scroll-sentinel");
  imageGrid.insertBefore(frag, sentinel);
  state.displayedCount = end;

  // Hide sentinel when all results are shown
  if (state.displayedCount >= state.allResults.length && sentinel) {
    sentinel.style.display = "none";
  }
}

/* IntersectionObserver for infinite scroll */
let scrollObserver = null;

function setupScrollObserver() {
  if (scrollObserver) scrollObserver.disconnect();

  const sentinel = document.getElementById("scroll-sentinel");
  if (!sentinel) return;

  sentinel.style.display = "";
  scrollObserver = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) appendNextPage();
  }, { rootMargin: "200px" });
  scrollObserver.observe(sentinel);
}

function renderGrid(results) {
  state.allResults    = results;
  state.displayedCount = 0;

  imageGrid.innerHTML = "";
  // Force reflow to re-trigger fadeIn animation
  void imageGrid.offsetWidth;
  imageGrid.style.animation = "none";

  // Add sentinel at the bottom
  const sentinel = document.createElement("div");
  sentinel.id = "scroll-sentinel";
  sentinel.style.height = "1px";
  imageGrid.appendChild(sentinel);

  requestAnimationFrame(() => {
    imageGrid.style.animation = "";
    appendNextPage();
    setupScrollObserver();
  });
}

function addToHistory(asin, text, category) {
  // Arrow between steps
  if (state.searchHistory.length > 0) {
    const arrow = document.createElement("div");
    arrow.className = "history-arrow";
    arrow.textContent = "↓";
    historyList.appendChild(arrow);
  }

  const item = document.createElement("div");
  item.className = "history-item";

  const thumb = document.createElement("img");
  thumb.className = "history-thumb";
  thumb.src = `/image/${asin}`;
  thumb.alt = asin;
  thumb.onerror = () => { thumb.style.background = "#e0e0e0"; thumb.removeAttribute("src"); };

  const textWrap = document.createElement("div");
  textWrap.className = "history-text";
  textWrap.innerHTML = `<strong>${category || "all"}</strong>${text}`;

  item.appendChild(thumb);
  item.appendChild(textWrap);
  historyList.appendChild(item);

  // Scroll history to bottom
  historyList.scrollTop = historyList.scrollHeight;

  state.searchHistory.push({ asin, text, category });
}

function openRefineModal(asin) {
  state.referenceAsin = asin;
  modalImg.src = `/image/${asin}`;
  showModal();
}

/* ── Initial text search ─────────────────────────────────── */
async function doInitialSearch() {
  const text     = document.getElementById("initial-text").value.trim();
  const category = document.getElementById("initial-cat").value;
  if (!text) return;

  showLoading();
  try {
    const res  = await fetch("/api/search", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, category }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    // Reset history
    historyList.innerHTML = "";
    state.searchHistory   = [];
    state.referenceAsin   = null;

    // First history entry: the text query itself
    const firstItem = document.createElement("div");
    firstItem.className = "history-item";
    firstItem.innerHTML = `
      <div class="history-text">
        <strong>${category || "all"}</strong>
        "${text}"
      </div>`;
    historyList.appendChild(firstItem);

    resultsLabel.textContent = `"${text}"`;
    renderGrid(data.results);
    showResults();
  } catch (e) {
    alert("Search failed: " + e.message);
  } finally {
    hideLoading();
  }
}

/* ── TIRG refine ─────────────────────────────────────────── */
async function doRefine() {
  const text     = refineText.value.trim();
  const category = refineCat.value;
  if (!text || !state.referenceAsin) return;

  const refAsin = state.referenceAsin;
  const refCat  = category || document.querySelector(`.img-card[data-asin="${refAsin}"]`)?.dataset.category || "";

  hideModal();
  showLoading();

  try {
    const res  = await fetch("/api/refine", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ reference_asin: refAsin, text, category }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    addToHistory(refAsin, text, refCat);
    resultsLabel.textContent = `"${text}"`;
    renderGrid(data.results);
  } catch (e) {
    alert("Refinement failed: " + e.message);
  } finally {
    hideLoading();
  }
}

/* ── Event listeners ─────────────────────────────────────── */
document.getElementById("search-btn").addEventListener("click", doInitialSearch);
document.getElementById("initial-text").addEventListener("keydown", e => {
  if (e.key === "Enter") doInitialSearch();
});

document.getElementById("refine-btn").addEventListener("click", doRefine);
document.getElementById("refine-text").addEventListener("keydown", e => {
  if (e.key === "Enter") doRefine();
});

document.getElementById("modal-close").addEventListener("click", hideModal);
modalOverlay.addEventListener("click", e => {
  if (e.target === modalOverlay) hideModal();
});

document.getElementById("new-search-btn").addEventListener("click", () => {
  resultsScreen.classList.remove("active");
  searchScreen.classList.add("active");
  document.getElementById("initial-text").value = "";
});
