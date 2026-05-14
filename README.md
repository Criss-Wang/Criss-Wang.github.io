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
