#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";

const sourcesPath = new URL("../src/data/live-sources.json", import.meta.url);
const outputPath = new URL("../src/data/live-stats.json", import.meta.url);

const sources = JSON.parse(await readFile(sourcesPath, "utf8"));
let existing = { generatedAt: null, publications: {}, repositories: {} };

try {
  existing = JSON.parse(await readFile(outputPath, "utf8"));
} catch {
  // First run; the generated file will be created below.
}

const headers = {
  Accept: "application/vnd.github+json",
  "User-Agent": "Criss-Wang.github.io live stats",
};

if (process.env.GITHUB_TOKEN) {
  headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
}

const semanticHeaders = {
  Accept: "application/json",
  "Content-Type": "application/json",
  "User-Agent": "Criss-Wang.github.io live stats",
};

if (process.env.SEMANTIC_SCHOLAR_API_KEY) {
  semanticHeaders["x-api-key"] = process.env.SEMANTIC_SCHOLAR_API_KEY;
}

const fetchJson = async (url, init = {}) => {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}`);
  }
  return response.json();
};

const githubJson = (path) => fetchJson(`https://api.github.com${path}`, { headers });

const commitCountFor = async (owner, repo) => {
  const response = await fetch(`https://api.github.com/repos/${owner}/${repo}/commits?per_page=1`, { headers });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${owner}/${repo} commits`);
  }

  const link = response.headers.get("link") ?? "";
  const lastPage = link.match(/[?&]page=(\d+)>;\s*rel="last"/);
  if (lastPage) {
    return Number(lastPage[1]);
  }

  const commits = await response.json();
  return Array.isArray(commits) ? commits.length : null;
};

const refreshRepositories = async () => {
  const refreshed = {};

  for (const [title, source] of Object.entries(sources.repositories ?? {})) {
    const fallback = existing.repositories?.[title] ?? {};
    try {
      const repo = await githubJson(`/repos/${source.owner}/${source.repo}`);
      let commitCount = fallback.commits ?? null;
      try {
        commitCount = await commitCountFor(source.owner, source.repo);
      } catch (error) {
        console.warn(`Keeping previous commit count for ${title}: ${error.message}`);
      }

      refreshed[title] = {
        name: repo.name,
        fullName: repo.full_name,
        url: repo.html_url,
        stars: repo.stargazers_count,
        forks: repo.forks_count,
        commits: commitCount,
        pushedAt: repo.pushed_at,
        updatedAt: repo.updated_at,
        defaultBranch: repo.default_branch,
        source: "GitHub",
      };
    } catch (error) {
      console.warn(`Keeping previous GitHub stats for ${title}: ${error.message}`);
      refreshed[title] = fallback;
    }
  }

  return refreshed;
};

const refreshPublications = async () => {
  const entries = Object.entries(sources.publications ?? {});
  const refreshed = {};

  if (entries.length === 0) {
    return refreshed;
  }

  try {
    const ids = entries.map(([, source]) => source.semanticScholarId);
    const response = await fetch(
      "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,citationCount,externalIds",
      {
        method: "POST",
        headers: semanticHeaders,
        body: JSON.stringify({ ids }),
      },
    );

    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText} from Semantic Scholar`);
    }

    const papers = await response.json();
    entries.forEach(([title], index) => {
      const paper = papers[index];
      const fallback = existing.publications?.[title] ?? {};
      refreshed[title] = paper
        ? {
            citationCount: paper.citationCount,
            source: "Semantic Scholar",
            sourceTitle: paper.title,
            externalIds: paper.externalIds ?? {},
          }
        : fallback;
    });
  } catch (error) {
    console.warn(`Keeping previous citation stats: ${error.message}`);
    entries.forEach(([title]) => {
      refreshed[title] = existing.publications?.[title] ?? {};
    });
  }

  return refreshed;
};

const liveStats = {
  generatedAt: new Date().toISOString(),
  publications: await refreshPublications(),
  repositories: await refreshRepositories(),
};

await writeFile(outputPath, `${JSON.stringify(liveStats, null, 2)}\n`);
console.log(`Updated ${outputPath.pathname}`);
