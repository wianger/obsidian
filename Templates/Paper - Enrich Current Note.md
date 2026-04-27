<%*
const vaultRoot = app.vault.adapter.basePath;
const activeFile = app.workspace.getActiveFile();
if (!activeFile) {
  new Notice("没有当前笔记。请先打开一篇论文主笔记。");
  return;
}

const commandPath = `${vaultRoot}/.obsidian/paper_enrich.js`;
const { execFile } = require("child_process");

await new Promise((resolve) => {
  execFile(
    "node",
    [commandPath, "--vault", vaultRoot, "--note", activeFile.path],
    { cwd: vaultRoot, windowsHide: true, maxBuffer: 1024 * 1024 * 32 },
    (error, stdout, stderr) => {
      if (stdout?.trim()) console.log(stdout.trim());
      if (stderr?.trim()) console.warn(stderr.trim());
      if (error) {
        new Notice(`摘要和标签处理失败：${error.message}`, 10000);
      } else {
        new Notice("论文摘要已翻译，标签已生成。");
      }
      resolve();
    }
  );
});
%>
