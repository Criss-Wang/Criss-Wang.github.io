export const homeHero = {
  roles: ["ML Research Engineer", "Vibe Researching", "Sunnyvale, CA"],
  intro:
    "ML engineer working across model iteration, machine learning systems, and AI-powered software. Long-running interest in bandits, reinforcement learning, OOD detection, and the practical infrastructure that makes production models tractable.",
  topics: ["ML systems", "model iteration", "OOD detection", "bandits", "AI software"],
};

export const homeProjectMeta: Record<
  string,
  {
    status: "active" | "wip" | "archived";
    repoTitle?: string;
    repoUrl?: string;
    stars?: number;
    pushedAt?: string;
    featuredTags?: string[];
    links?: Array<{ label: string; href: string }>;
  }
> = {
  "Needle: High-performance DL System": {
    status: "active",
    repoTitle: "needle",
    repoUrl: "https://github.com/Criss-Wang/needle",
    stars: 0,
    pushedAt: "2023-12-13T04:08:54Z",
    featuredTags: ["c++", "python", "cuda"],
    links: [{ label: "code", href: "https://github.com/Criss-Wang/needle" }],
  },
  "Motion Prediction with Guided Diffusion": {
    status: "archived",
    repoTitle: "trajectory-diffusion",
    repoUrl: "https://github.com/Criss-Wang/trajectory-diffusion",
    stars: 0,
    pushedAt: "2023-04-29T01:54:37Z",
    featuredTags: ["pytorch", "diffusion", "autonomous vehicle"],
    links: [
      { label: "code", href: "https://github.com/Criss-Wang/trajectory-diffusion" },
      {
        label: "report",
        href: "https://drive.google.com/file/d/118t8mAokTr4-YEQ5pSTlT4QPrGgBUzN-/view?usp=drive_link",
      },
    ],
  },
  "Starlink Tracking": {
    status: "archived",
    repoTitle: "Satellite-Tracker",
    repoUrl: "https://github.com/Criss-Wang/Satellite-Tracker",
    stars: 0,
    pushedAt: "2022-12-01T02:40:18Z",
    featuredTags: ["d3.js", "javascript", "react"],
    links: [{ label: "code", href: "https://github.com/Criss-Wang/Satellite-Tracker" }],
  },
  "Travel Planner": {
    status: "archived",
    repoTitle: "Travel-Planner",
    repoUrl: "https://github.com/Criss-Wang/Travel-Planner",
    stars: 0,
    pushedAt: "2023-01-02T23:12:57Z",
    featuredTags: ["gpt", "go", "mysql"],
    links: [
      { label: "code", href: "https://github.com/Criss-Wang/Travel-Planner" },
      {
        label: "slides",
        href: "https://docs.google.com/presentation/d/1OWBaqNC8_jkJSOGyLIuRiuyYzgNggsb9lkeuI2QteJw/edit?usp=sharing",
      },
    ],
  },
};
