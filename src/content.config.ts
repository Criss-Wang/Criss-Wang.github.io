import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const stringList = z.union([z.string(), z.array(z.string())]).optional();
const looseBoolean = z
  .union([z.boolean(), z.string().transform((value) => value.toLowerCase() === "true")])
  .optional();

const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z
    .object({
      title: z.string(),
      date: z.coerce.date(),
      updated: z.coerce.date().optional(),
      excerpt: z.string().optional(),
      categories: stringList,
      tags: stringList,
      layout: z.string().optional(),
      mathjax: looseBoolean,
      toc: looseBoolean,
      sticky: z.number().optional(),
      link: z.string().optional(),
      draft: z.boolean().optional(),
    })
    .passthrough(),
});

const reviewership = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/reviewership" }),
  schema: z.object({
    year: z.coerce.string(),
    venue: z.string(),
    order: z.number().optional(),
    draft: z.boolean().optional(),
  }),
});

export const collections = { posts, reviewership };
