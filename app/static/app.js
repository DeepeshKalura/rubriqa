/* Minimal vanilla JS: confirm dialogs and a tiny relative-time refresh. */
document.addEventListener("submit", function (event) {
  const form = event.target;
  const message = form.getAttribute("data-confirm");
  if (message && !window.confirm(message)) {
    event.preventDefault();
  }
});

document.addEventListener("click", function (event) {
  const link = event.target.closest("a[data-confirm]");
  if (link && !window.confirm(link.getAttribute("data-confirm"))) {
    event.preventDefault();
  }
});

function formatRelative(iso) {
  if (!iso) return "";
  const then = new Date(iso);
  const seconds = Math.floor((Date.now() - then.getTime()) / 1000);
  if (seconds < 60) return seconds + "s ago";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m ago";
  if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago";
  return Math.floor(seconds / 86400) + "d ago";
}

document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("[data-relative]").forEach(function (el) {
    const iso = el.getAttribute("data-relative");
    el.textContent = formatRelative(iso);
    el.title = iso;
  });
});
