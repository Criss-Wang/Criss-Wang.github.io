import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { excerptFor, postDate, postUrl, sortPosts } from "../lib/posts";
import { profile } from "../data/profile";

export async function GET(context: { site: URL }) {
  const posts = sortPosts(await getCollection("posts", ({ data }) => !data.draft));
  return rss({
    title: `${profile.name} - Writing`,
    description: profile.tagline,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: excerptFor(post),
      pubDate: postDate(post),
      link: postUrl(post),
    })),
  });
}
