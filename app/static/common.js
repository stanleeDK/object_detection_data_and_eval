// Helpers shared by all pages.

// One color per class id (class 0 = orange, 1 = blue, ...), reused if there are more classes.
const CLASS_COLORS = ["#f97316", "#3b82f6", "#22c55e", "#e11d48", "#a855f7", "#eab308", "#06b6d4", "#ec4899", "#84cc16"];

function classColor(cls) {
  return CLASS_COLORS[cls % CLASS_COLORS.length];
}

// Call one of the server's /api/... endpoints and return its JSON.
// On an error, throw the server's explanation (FastAPI puts it in "detail").
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try { message = (await res.json()).detail || message; } catch {}
    throw new Error(message);
  }
  return res.json();
}

// Make text safe to put inside HTML (so a file name like "<b>.jpg" shows as text).
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Load an image URL into an <img>, waiting until it's ready to draw.
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("The browser couldn't display this image."));
    img.src = src;
  });
}

// Draw one box with a colored name tag on its top edge. Corners are canvas pixels.
// dashed = a suggestion not yet accepted; selected = thicker outline + light fill.
function drawBox(ctx, { x1, y1, x2, y2 }, color, label, { dashed = false, selected = false, lineWidth = 2 } = {}) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? lineWidth + 1.5 : lineWidth;
  if (dashed) ctx.setLineDash([6, 4]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  if (selected) {
    ctx.fillStyle = color + "22";  // "22" = mostly transparent
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (label) {
    ctx.font = "600 12px -apple-system, BlinkMacSystemFont, sans-serif";
    const tagWidth = ctx.measureText(label).width + 8;
    const tagTop = y1 >= 18 ? y1 - 18 : y1;  // above the box, or inside it if there's no room
    ctx.fillStyle = color;
    ctx.fillRect(x1 - ctx.lineWidth / 2, tagTop, tagWidth, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 4 - ctx.lineWidth / 2, tagTop + 13);
  }
  ctx.restore();
}
