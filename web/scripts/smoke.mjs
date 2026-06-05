/* Headless smoke test: mount the built bundle in jsdom with the real
   engine data and assert each workflow renders without throwing.
   Run after `npm run build`:  node scripts/smoke.mjs                  */
import { JSDOM } from "jsdom";
import { readFileSync, readdirSync } from "node:fs";
import { pathToFileURL } from "node:url";

const data = readFileSync(new URL("../dist/data/console_data.json", import.meta.url), "utf8");
const bundle = readdirSync(new URL("../dist/assets/", import.meta.url)).find((f) => f.endsWith(".js"));
if (!bundle) throw new Error("no built JS bundle in dist/assets — run `npm run build` first");

const dom = new JSDOM(`<!DOCTYPE html><html><body><div id="root"></div></body></html>`, {
  url: "http://localhost/",
  pretendToBeVisual: true,
});
const { window } = dom;
globalThis.window = window;
globalThis.document = window.document;
globalThis.localStorage = window.localStorage;
for (const k of [
  "HTMLElement", "Node", "Element", "SVGElement", "Event", "MouseEvent",
  "CustomEvent", "MutationObserver", "DocumentFragment", "Text", "requestAnimationFrame",
  "cancelAnimationFrame",
]) {
  globalThis[k] = window[k];
}
globalThis.getComputedStyle = window.getComputedStyle.bind(window);

const errors = [];
const origError = console.error;
console.error = (...a) => {
  errors.push(a.join(" "));
  origError(...a);
};

globalThis.fetch = async () => ({
  ok: true,
  status: 200,
  json: async () => JSON.parse(data),
  text: async () => data,
});

const root = document.getElementById("root");

async function settle(ms = 60) {
  await new Promise((r) => setTimeout(r, ms));
}

function assert(cond, msg) {
  if (!cond) {
    console.error("ASSERT FAILED:", msg);
    process.exitCode = 1;
  } else {
    console.log("  ✓", msg);
  }
}

// click a nav button by its label text, then settle
async function clickNav(label) {
  const btn = [...document.querySelectorAll(".nav button")].find((b) =>
    b.textContent.trim().startsWith(label)
  );
  if (!btn) throw new Error("nav button not found: " + label);
  btn.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle();
}
async function clickTab(idx) {
  const tabs = [...document.querySelectorAll(".tabs button")];
  tabs[idx]?.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle();
}

await import(pathToFileURL(new URL(`../dist/assets/${bundle}`, import.meta.url).pathname));
await settle(120); // allow data fetch + first render

console.log("default (predict / Caffeine):");
assert(root.textContent.includes("Sisyphus"), "brand renders");
assert(root.textContent.includes("Caffeine"), "default drug Caffeine renders");
assert(root.textContent.includes("Cmax") || root.querySelector(".statcell"), "endpoint stats render");
assert(root.querySelector("svg.chart"), "concentration-time chart renders");
assert(root.textContent.includes("in domain"), "AD badge renders");

const WF = ["predict", "simulate", "tdm", "ddi", "dose-adjust", "benchmark"];
for (const w of WF) {
  await clickNav(w);
  const tabCount = document.querySelectorAll(".tabs button").length;
  let ok = true;
  for (let i = 0; i < tabCount; i++) {
    await clickTab(i);
    const hasContent = document.querySelector(".content").textContent.trim().length > 20;
    if (!hasContent) ok = false;
  }
  assert(ok && tabCount > 0, `workflow "${w}" renders all ${tabCount} tab(s)`);
}

// benchmark scatter has 107 points
await clickNav("benchmark");
await clickTab(0);
const circles = document.querySelectorAll(".content svg.chart circle");
assert(circles.length >= 100, `benchmark scatter renders ${circles.length} points (≥100)`);

const realErrors = errors.filter(
  (e) => !/Warning:|act\(|StrictMode|defaultProps|deprecated/i.test(e)
);
assert(realErrors.length === 0, `no runtime console errors (${realErrors.length})`);
if (realErrors.length) realErrors.forEach((e) => console.log("    !", e.slice(0, 200)));

console.log(process.exitCode ? "\nSMOKE: FAIL" : "\nSMOKE: PASS");
