/* Copy the built bundle (web/dist) into the repo-root /app folder, which the
   existing Jekyll Pages deployment serves at sisyphus-pbpk.io/app/ (the Jekyll
   homepage stays at /). Run via `npm run build:pages`. */
import { rmSync, mkdirSync, cpSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, "../dist");
const app = resolve(here, "../../app");

if (!existsSync(dist)) {
  console.error("web/dist not found — run `npm run build` first.");
  process.exit(1);
}
rmSync(app, { recursive: true, force: true });
mkdirSync(app, { recursive: true });
cpSync(dist, app, { recursive: true });
console.log(`synced ${dist} -> ${app}`);
