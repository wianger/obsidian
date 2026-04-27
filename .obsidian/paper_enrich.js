#!/usr/bin/env node

const fs = require("fs/promises");
const path = require("path");

const DEFAULT_LLM_CONFIG = {
  apiKey: "",
  baseUrl: "https://api.openai.com",
  model: "",
  temperature: 0.2,
};

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i];
    if (item.startsWith("--")) {
      const key = item.slice(2);
      const value = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
      args[key] = value;
    }
  }
  return args;
}

function decodeHtmlEntities(value) {
  return String(value || "")
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, code) => String.fromCharCode(parseInt(code, 16)))
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function yamlScalar(value) {
  if (value == null) return "";
  const trimmed = String(value).trim();
  if (!trimmed) return "";
  if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseFrontmatter(markdown) {
  const normalized = markdown.replace(/^\uFEFF/, "");
  const start = normalized.match(/^\s*---\r?\n/);
  if (!start) return {};
  const contentStart = start[0].length;
  const end = normalized.indexOf("\n---", contentStart);
  if (end === -1) return {};
  const block = normalized.slice(contentStart, end).split(/\r?\n/);
  const data = {};
  let currentKey = null;
  for (let index = 0; index < block.length; index += 1) {
    const line = block[index];
    const match = line.match(/^([^:]+):\s*(.*)$/);
    if (match) {
      currentKey = match[1].trim();
      const rawValue = match[2];
      if (/^[>|]-?$/.test(rawValue.trim())) {
        const lines = [];
        while (index + 1 < block.length && (/^\s/.test(block[index + 1]) || block[index + 1].trim() === "")) {
          index += 1;
          lines.push(block[index].replace(/^ {2}/, ""));
        }
        data[currentKey] = lines.join("\n").trimEnd();
      } else {
        data[currentKey] = yamlScalar(rawValue);
      }
      continue;
    }
    if (currentKey && /^\s+-\s+/.test(line)) {
      const existing = Array.isArray(data[currentKey]) ? data[currentKey] : [];
      existing.push(yamlScalar(line.replace(/^\s+-\s+/, "")));
      data[currentKey] = existing;
    }
  }
  return data;
}

function frontmatterBounds(markdown) {
  const normalized = markdown.replace(/^\uFEFF/, "");
  const start = normalized.match(/^\s*---\r?\n/);
  if (!start) return null;
  const contentStart = start[0].length;
  const end = normalized.indexOf("\n---", contentStart);
  if (end === -1) return null;
  return { start: 0, end: end + 4 };
}

function formatFrontmatterEntry(key, value, options = {}) {
  if (options.list) {
    const values = Array.isArray(value) ? value : [];
    const items = values
      .map((item) => String(item || "").trim())
      .filter(Boolean)
      .map((item) => `"${item.replace(/"/g, '\\"')}"`);
    return `${key}: [${items.join(", ")}]`;
  }
  const text = String(value || "").trimEnd();
  if (options.block || text.includes("\n")) {
    const body = text
      ? text.split(/\r?\n/).map((line) => `  ${line}`).join("\n")
      : "  ";
    return `${key}: |-\n${body}`;
  }
  const escaped = text.replace(/"/g, '\\"');
  return `${key}: "${escaped}"`;
}

function setFrontmatterValue(markdown, key, value, options = {}) {
  const bounds = frontmatterBounds(markdown);
  if (!bounds) return markdown;
  const before = markdown.slice(0, bounds.end);
  const after = markdown.slice(bounds.end);
  const lines = before.split(/\r?\n/);
  const entry = formatFrontmatterEntry(key, value, options).split("\n");
  const start = lines.findIndex((line) => line.startsWith(`${key}:`));
  if (start >= 0) {
    let end = start + 1;
    while (end < lines.length - 1 && (/^\s/.test(lines[end]) || lines[end].trim() === "")) {
      end += 1;
    }
    lines.splice(start, end - start, ...entry);
  } else {
    lines.splice(lines.length - 1, 0, ...entry);
  }
  return `${lines.join("\n")}${after}`;
}

function formatReadableTimestamp(date = new Date()) {
  const pad = (value, size = 2) => String(value).padStart(size, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function getMetaValue(meta, keys) {
  for (const key of keys) {
    const value = meta[key];
    if (value == null) continue;
    const text = Array.isArray(value) ? value.join("; ") : String(value);
    if (text.trim()) return decodeHtmlEntities(text).trim();
  }
  return "";
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function numberOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

async function readJsonIfExists(filePath) {
  if (!(await fileExists(filePath))) return {};
  const raw = await fs.readFile(filePath, "utf8");
  try {
    return JSON.parse(raw);
  } catch (error) {
    error.message = `Invalid JSON in ${filePath}: ${error.message}`;
    throw error;
  }
}

async function loadLlmConfig(vault) {
  const configPath = path.join(vault, ".obsidian", "llm.json");
  const fileConfig = await readJsonIfExists(configPath);
  return {
    configPath,
    apiKey: fileConfig.apiKey || DEFAULT_LLM_CONFIG.apiKey,
    baseUrl: fileConfig.baseUrl || DEFAULT_LLM_CONFIG.baseUrl,
    model: fileConfig.model || DEFAULT_LLM_CONFIG.model,
    temperature: numberOrDefault(fileConfig.temperature, DEFAULT_LLM_CONFIG.temperature),
  };
}

async function chat(messages, { baseUrl, apiKey, model, temperature }) {
  const endpoint = `${baseUrl.replace(/\/$/, "")}/v1/chat/completions`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`LLM request failed ${response.status}: ${body.slice(0, 1000)}`);
  }

  const json = await response.json();
  const content = json.choices?.[0]?.message?.content;
  if (!content) throw new Error("LLM response did not include message content.");
  return content.trim();
}

function normalizeTranslation(text) {
  return String(text || "")
    .replace(/^```(?:markdown)?\s*/i, "")
    .replace(/```\s*$/i, "")
    .replace(/\r?\n+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseJsonArrayLike(text) {
  const normalized = String(text || "")
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/```\s*$/i, "")
    .trim();

  const candidates = [normalized];
  const arrayMatch = normalized.match(/\[[\s\S]*\]/);
  if (arrayMatch) candidates.push(arrayMatch[0]);

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (Array.isArray(parsed)) return parsed;
      if (Array.isArray(parsed.tags)) return parsed.tags;
    } catch {
      // Try the next candidate, then fall back to delimiter parsing below.
    }
  }

  return normalized
    .split(/[\n,，;；]+/)
    .map((item) => item.replace(/^[-*\d.\s]+/, "").trim())
    .filter(Boolean);
}

function normalizeTag(value) {
  return decodeHtmlEntities(value)
    .normalize("NFKC")
    .trim()
    .replace(/^#+/, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[()[\]{}"'`.,:;!?]/g, " ")
    .replace(/[\\/]+/g, "-")
    .replace(/[_\s]+/g, "-")
    .replace(/[^a-z0-9\u4e00-\u9fff-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
}

function uniqueTags(values, limit = 8) {
  const seen = new Set();
  const tags = [];
  for (const value of values) {
    const tag = normalizeTag(value);
    if (!tag || seen.has(tag)) continue;
    seen.add(tag);
    tags.push(tag);
    if (tags.length >= limit) break;
  }
  return tags;
}

async function translateAbstract(meta, llmConfig, abstract) {
  const { apiKey, baseUrl, model, temperature, configPath } = llmConfig;

  if (!apiKey || !model) {
    throw new Error(`摘要翻译未运行：请先编辑配置文件 ${configPath}，填写 apiKey、baseUrl 和 model。`);
  }

  if (!abstract) {
    throw new Error("找不到 Zotero 摘要：当前笔记 frontmatter 中 abstract 为空。");
  }

  const translated = await chat([
    {
      role: "system",
      content: "你是严谨的学术翻译助手。只翻译用户提供的论文摘要，不补充、不总结、不扩写、不解释。输出自然准确的中文，保留必要的英文术语、缩写、模型名、数据集名和方法名。",
    },
    {
      role: "user",
      content: [
        `论文标题：${meta.title || ""}`,
        "",
        "请将下面的 Zotero 摘要翻译为中文，只输出译文。不要添加标题、编号、列表、解释或额外总结。译文保持为一段话。",
        "",
        abstract,
      ].join("\n"),
    },
  ], { baseUrl, apiKey, model, temperature });

  return {
    model,
    text: normalizeTranslation(translated),
  };
}

async function generateTags(meta, llmConfig, abstract) {
  const { apiKey, baseUrl, model, temperature, configPath } = llmConfig;

  if (!apiKey || !model) {
    throw new Error(`标签生成未运行：请先编辑配置文件 ${configPath}，填写 apiKey、baseUrl 和 model。`);
  }

  if (!abstract) {
    throw new Error("标签生成失败：当前笔记 frontmatter 中 abstract 为空。");
  }

  const raw = await chat([
    {
      role: "system",
      content: "你是论文阅读笔记的标签整理助手。你只输出 JSON 数组，不输出解释、标题或 Markdown。",
    },
    {
      role: "user",
      content: [
        `论文标题：${meta.title || ""}`,
        `发表载体：${meta.venue || ""}`,
        "",
        "请根据下面的论文摘要生成 5 到 8 个适合 Obsidian frontmatter 的英文标签。",
        "要求：",
        "- 只输出 JSON 字符串数组，例如 [\"llm\", \"kv-cache\", \"model-serving\"]",
        "- 标签使用小写英文短语，多个词用连字符",
        "- 不要添加 #",
        "- 优先覆盖研究对象、方法、系统/任务、关键技术",
        "- 不要输出过宽泛的标签，例如 paper、research、computer-science",
        "",
        abstract,
      ].join("\n"),
    },
  ], { baseUrl, apiKey, model, temperature });

  const tags = uniqueTags(parseJsonArrayLike(raw));
  if (!tags.length) throw new Error("标签生成失败：LLM 未返回可用标签。");
  return tags;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const vault = path.resolve(args.vault || process.cwd());
  const noteRel = String(args.note || "").replace(/\\/g, "/");
  if (!noteRel) throw new Error("Missing --note <vault-relative note path>.");

  const notePath = path.join(vault, noteRel);
  let markdown = await fs.readFile(notePath, "utf8");
  const meta = parseFrontmatter(markdown);

  if (!meta.created) {
    const created = formatReadableTimestamp();
    markdown = setFrontmatterValue(markdown, "created", created);
    meta.created = created;
  }

  const llmConfig = await loadLlmConfig(vault);
  const abstract = getMetaValue(meta, ["abstract"]);
  const result = await translateAbstract(meta, llmConfig, abstract);
  const tags = await generateTags(meta, llmConfig, abstract);
  markdown = setFrontmatterValue(markdown, "abstract", result.text, { block: true });
  markdown = setFrontmatterValue(markdown, "tags", tags, { list: true });

  await fs.writeFile(notePath, markdown, "utf8");
  console.log(JSON.stringify({ note: noteRel, abstract: "translated", tags }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
