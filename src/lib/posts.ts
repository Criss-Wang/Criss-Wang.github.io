import type { CollectionEntry } from "astro:content";

export type Post = CollectionEntry<"posts">;

export function listify(value: string | string[] | undefined): string[] {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

export function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function postSlug(post: Post): string {
  return post.id
    .split("/")
    .map((part) => slugify(part.replace(/\.md$/i, "")))
    .filter(Boolean)
    .join("/");
}

export function postUrl(post: Post): string {
  return `/writing/${postSlug(post)}/`;
}

export function postDate(post: Post): Date {
  return post.data.date instanceof Date ? post.data.date : new Date(post.data.date);
}

export function sortPosts(posts: Post[]): Post[] {
  return [...posts].sort((a, b) => postDate(b).getTime() - postDate(a).getTime());
}

export function formatDate(date: Date): string {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `${year}.${month}.${day}`;
}

export function formatIsoDate(date: Date): string {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function readingMinutes(body = ""): number {
  const text = body
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[^\w\s'-]/g, " ");
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 220));
}

export function excerptFor(post: Post): string {
  if (post.data.excerpt) return post.data.excerpt;
  return `${post.body
    .replace(/^---[\s\S]*?---/, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/[#*_`>\-\[\]\(\)!]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 180)}...`;
}

export function postTags(post: Post): string[] {
  return listify(post.data.tags);
}

export function postCategories(post: Post): string[] {
  return listify(post.data.categories);
}

export function isProjectPost(post: Post): boolean {
  return postCategories(post).some((category) => category.toLowerCase() === "projects");
}

export function searchText(post: Post): string {
  return [
    post.data.title,
    excerptFor(post),
    ...postTags(post),
    ...postCategories(post),
    postSlug(post),
  ]
    .join(" ")
    .toLowerCase();
}
