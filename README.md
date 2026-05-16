# Criss Wang's Log Book

This site is built with Astro. The previous Hexo/Icarus setup is no longer used by the active build.

## Content

- Blog posts: `src/content/posts/**/*.md`
- Personal profile and links: `src/data/profile.ts`
- Homepage `/now`: `src/data/now.ts`
- Publications: `src/data/publications.ts`
- Images: `public/images`

Markdown posts keep YAML front matter for `title`, `date`, `updated`, `categories`, `tags`, `excerpt`, `mathjax`, and related metadata.

## Dev

- `npm ci`
- `npm run dev`

## Prod

- `npm run build`

## Analytics

Production analytics are wired for Cloudflare Web Analytics, but disabled unless a token is provided at build time.

1. In Cloudflare, add Web Analytics for `criss-wang.com`.
2. Copy only the beacon token from the Cloudflare snippet.
3. In GitHub, add an Actions secret named `CF_WEB_ANALYTICS_TOKEN`.
4. Push or manually rerun the Pages workflow.

The deployed site will then record page-level traffic, including individual post URLs under `/writing/...`. Local development and builds without the secret do not emit the analytics script.
