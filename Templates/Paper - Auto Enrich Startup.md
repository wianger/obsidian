<%*
const vaultRoot = app.vault.adapter.basePath;
const enrichScriptPath = `${vaultRoot}/.obsidian/paper_enrich.js`;
const { execFile } = require("child_process");

if (!window.paperAutoEnrichState) {
  window.paperAutoEnrichState = {
    running: new Set(),
    registered: false,
    ref: null,
  };
}

const state = window.paperAutoEnrichState;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function shouldHandle(file) {
  return (
    file?.extension === "md" &&
    file.path.startsWith("Papers/") &&
    !file.path.startsWith("Templates/")
  );
}

async function waitUntilImportLooksReady(file, timeoutMs = 30000) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    const content = await app.vault.cachedRead(file);
    if (/^---\r?\n[\s\S]*?\r?\n---/.test(content) && /^abstract:/m.test(content)) {
      return true;
    }
    await sleep(500);
  }

  return false;
}

async function enrichPaper(file) {
  if (!shouldHandle(file) || state.running.has(file.path)) return;
  state.running.add(file.path);

  try {
    const ready = await waitUntilImportLooksReady(file);
    if (!ready) {
      new Notice(`论文自动处理跳过：${file.basename} 还没有可用 abstract。`, 10000);
      return;
    }

    new Notice(`正在处理论文摘要和标签：${file.basename}`, 8000);

    await new Promise((resolve, reject) => {
      execFile(
        "node",
        [enrichScriptPath, "--vault", vaultRoot, "--note", file.path],
        { cwd: vaultRoot, windowsHide: true, maxBuffer: 1024 * 1024 * 32 },
        (error, stdout, stderr) => {
          if (stdout?.trim()) console.log(stdout.trim());
          if (stderr?.trim()) console.warn(stderr.trim());
          if (error) {
            reject(error);
          } else {
            resolve();
          }
        }
      );
    });

    new Notice(`论文摘要和标签已处理：${file.basename}`, 8000);
  } catch (error) {
    console.error(error);
    new Notice(`论文自动处理失败：${error.message}`, 10000);
  } finally {
    state.running.delete(file.path);
  }
}

if (!state.registered) {
  state.ref = app.vault.on("create", (file) => {
    if (shouldHandle(file)) {
      enrichPaper(file);
    }
  });
  state.registered = true;
  new Notice("论文自动处理已启用：Zotero 导入到 Papers/ 后会自动翻译摘要并生成标签。", 8000);
}
%>
