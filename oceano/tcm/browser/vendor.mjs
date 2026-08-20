#!/usr/bin/env node
/**
 * Browser-runtime dependency automation — npm is the sole resolver/lockfile.
 *
 * This directory is the whole browser subsystem chain:
 *
 *     package.json + package-lock.json   committed declaration & lock
 *     node_modules/                      npm ci output (generated)
 *     _build/browser-runtime/            served runtime (generated)
 *     src/tcm_gui/browser/server.py      serves it under /assets/
 *
 * Commands (pixi tasks for sync/update; lock/clean call the script directly):
 *
 *     node browser/vendor.mjs lock     npm install --package-lock-only
 *     node browser/vendor.mjs sync     npm ci + rebuild _build/browser-runtime
 *     node browser/vendor.mjs update   npm i --save-exact <deps>@latest + sync
 *     node browser/vendor.mjs clean    rm node_modules _build/browser-runtime
 *
 * File map (node_modules → _build/browser-runtime/) — the minimal set the
 * viewer requests under /assets/; the whole MathJax package is NOT copied
 * (~8 MB runtime vs ~50 MB package):
 *
 *     marked/marked.min.js                    marked lib/marked.umd.js (UMD, global marked)
 *     mathjax/tex-chtml.js                    mathjax combined TeX→CHTML component (package root)
 *     mathjax/sre/                            mathjax sre/ — speech worker + mathmaps (a11y)
 *     mathjax/output/fonts/mathjax-newcm/     @mathjax/mathjax-newcm-font package root: chtml.js
 *                                             stub + chtml/woff2 (font faces) + chtml/dynamic (lazy chunks)
 *     highlight/highlight.min.js              cdnjs build pinned to the locked version — the npm
 *                                             package ships no browser bundle (lib/ is CommonJS)
 *     highlight/default.min.css               highlight.js styles/
 *     each package's LICENSE file          under its runtime subdirectory
 *
 * MathJax constraints (learned the hard way):
 * 1. The output/fonts/mathjax-newcm/ tree must keep this exact shape — it
 *    mirrors the loader path mapping the stub installs.  Do not flatten.
 * 2. The font-package stub (chtml.js) must be vendored AND
 *    loader.paths["mathjax-newcm"] must point at it in index.html —
 *    otherwise the loader resolves the unknown package to its jsdelivr CDN
 *    default and fonts load from the Internet.
 *
 * Licenses: marked — MIT, MathJax — Apache-2.0, highlight.js — BSD-3-Clause.
 */
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const TCM_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const BROWSER = join(TCM_ROOT, "browser");
const nm = (...p) => join(BROWSER, "node_modules", ...p);
const VENDOR = join(TCM_ROOT, "_build", "browser-runtime");
const NPM = process.platform === "win32" ? "npm.cmd" : "npm";

// node_modules source → _build/browser-runtime destination (relative)
const FILES = [
    [nm("marked", "lib", "marked.umd.js"), "marked/marked.min.js"],
    [nm("marked", "LICENSE"), "marked/LICENSE"],
    [nm("mathjax", "LICENSE"), "mathjax/LICENSE"],
    [nm("mathjax", "tex-chtml.js"), "mathjax/tex-chtml.js"],
    [nm("@mathjax", "mathjax-newcm-font", "chtml.js"), "mathjax/output/fonts/mathjax-newcm/chtml.js"],
    [nm("highlight.js", "styles", "default.min.css"), "highlight/default.min.css"],
    [nm("highlight.js", "LICENSE"), "highlight/LICENSE"],
];
const DIRS = [
    [nm("mathjax", "sre"), "mathjax/sre"],  // speech-rule-engine worker + mathmaps (a11y)
    [nm("@mathjax", "mathjax-newcm-font", "chtml"), "mathjax/output/fonts/mathjax-newcm/chtml"],  // woff2 + dynamic chunks
];

const run = (cmd, args) => {
    console.log(`[browser] ${cmd} ${args.join(" ")}`);
    // shell: win32 .cmd shims require it (spawn *.cmd EINVAL since Node's CVE-2024-27980 fix)
    execFileSync(cmd, args, { cwd: BROWSER, stdio: "inherit", shell: process.platform === "win32" });
};

const npmInstall = (...extra) =>
    run(NPM, ["install", ...extra, "--ignore-scripts", "--no-audit", "--no-fund"]);

// npm ci: exact locked tree, wipes node_modules first, never touches the manifests
const npmCi = () => run(NPM, ["ci", "--ignore-scripts", "--no-audit", "--no-fund"]);

function treeStats(p) {
    let nFiles = 0, nBytes = 0;
    for (const e of readdirSync(p, { withFileTypes: true })) {
        const q = join(p, e.name);
        if (e.isDirectory()) { const [fn, fb] = treeStats(q); nFiles += fn; nBytes += fb; }
        else { nFiles++; nBytes += statSync(q).size; }
    }
    return [nFiles, nBytes];
}

async function sync() {
    rmSync(VENDOR, { recursive: true, force: true });  // drop stale files of any older runtime
    npmCi();

    const absent = [...FILES, ...DIRS].map(([s]) => s).filter((s) => !existsSync(s));
    if (absent.length) throw new Error(`Upstream layout changed — update the file map; missing:\n  ${absent.join("\n  ")}`);

    for (const [src, dest] of FILES) {
        mkdirSync(dirname(join(VENDOR, dest)), { recursive: true });
        cpSync(src, join(VENDOR, dest));
    }
    for (const [src, dest] of DIRS) cpSync(src, join(VENDOR, dest), { recursive: true });

    // Browser bundle absent from npm — cdnjs build of the exact locked version
    const hljsVer = JSON.parse(readFileSync(nm("highlight.js", "package.json"), "utf8")).version;
    const url = `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/${hljsVer}/highlight.min.js`;
    console.log(`[browser] fetch ${url}`);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`cdnjs ${r.status} for highlight.min.js ${hljsVer}`);
    writeFileSync(join(VENDOR, "highlight", "highlight.min.js"), Buffer.from(await r.arrayBuffer()));

    const [nFiles, nBytes] = treeStats(VENDOR);
    console.log(`[browser] runtime synced: ${nFiles} files, ${(nBytes / 1024).toFixed(1)} KiB → ${VENDOR}`);
}

const lock = () => npmInstall("--package-lock-only");

const update = () => {
    const deps = Object.keys(JSON.parse(readFileSync(join(BROWSER, "package.json"), "utf8")).dependencies);
    npmInstall("--save-exact", ...deps.map((d) => `${d}@latest`));
    sync();
};

const clean = () => {
    rmSync(join(BROWSER, "node_modules"), { recursive: true, force: true });
    rmSync(VENDOR, { recursive: true, force: true });
    console.log("[browser] node_modules and _build/browser-runtime removed");
};

switch (process.argv[2] ?? "sync") {
    case "lock": lock(); break;
    case "sync": sync(); break;
    case "update": update(); break;
    case "clean": clean(); break;
    default: console.error("Usage: browser_vendor.mjs {lock|sync|update|clean}"); process.exitCode = 2;
}
