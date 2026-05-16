export const homeHero = {
  roles: ["Machine Learning Engineer", "Agentic AI Systems", "Sunnyvale, CA"],
  intro:
    "Machine learning engineer focused on production reasoning engines, agent memory, and event-driven RL. I turn ML research into deployable agentic systems: tool use, orchestration, evaluation, and robustness when conditions shift.",
  topics: ["Agent Memory", "Event-driven RL", "Reasoning Engine", "ML Robustness"],
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
    showOnProjects?: boolean;
    links?: Array<{ label: string; href: string }>;
  }
> = {
  "Heron Event-Driven MARL": {
    status: "active",
    repoTitle: "HERON",
    featuredTags: ["multi-agent rl", "event-driven", "simulation"],
  },
  "LLM Validator": {
    status: "active",
    repoTitle: "llm-validator",
    repoUrl: "https://github.com/Criss-Wang/llm-validator",
    stars: 0,
    pushedAt: "2024-09-20T21:23:11Z",
    featuredTags: ["llm evaluation", "model validation", "benchmarking"],
    links: [
      { label: "code", href: "https://github.com/Criss-Wang/llm-validator" },
      { label: "writeup", href: "/writing/software/model-iteration-research-validation/" },
    ],
  },
  ResumeAssist: {
    status: "wip",
    repoTitle: "ResumeAssist",
    repoUrl: "https://github.com/Criss-Wang/ResumeAssist",
    stars: 1,
    pushedAt: "2024-10-24T22:56:42Z",
    featuredTags: ["agentic ai", "langchain", "full-stack ai"],
    links: [{ label: "code", href: "https://github.com/Criss-Wang/ResumeAssist" }],
  },
  "Deployable AI": {
    status: "archived",
    repoTitle: "dpai",
    repoUrl: "https://github.com/Criss-Wang/dpai",
    stars: 1,
    pushedAt: "2024-06-23T18:05:45Z",
    featuredTags: ["model serving", "mlops", "fastapi"],
    links: [
      { label: "code", href: "https://github.com/Criss-Wang/dpai" },
      { label: "docs", href: "https://dpai.readthedocs.io/" },
    ],
  },
  "Needle: High-performance DL System": {
    status: "active",
    repoTitle: "needle",
    repoUrl: "https://github.com/Criss-Wang/needle",
    stars: 0,
    pushedAt: "2023-12-13T04:08:54Z",
    featuredTags: ["deep learning", "autodiff", "cuda"],
    links: [{ label: "code", href: "https://github.com/Criss-Wang/needle" }],
  },
  "Motion Prediction with Guided Diffusion": {
    status: "archived",
    repoTitle: "trajectory-diffusion",
    repoUrl: "https://github.com/Criss-Wang/trajectory-diffusion",
    stars: 0,
    pushedAt: "2023-04-29T01:54:37Z",
    featuredTags: ["diffusion models", "trajectory prediction", "pytorch"],
    links: [
      { label: "code", href: "https://github.com/Criss-Wang/trajectory-diffusion" },
      {
        label: "report",
        href: "https://drive.google.com/file/d/118t8mAokTr4-YEQ5pSTlT4QPrGgBUzN-/view?usp=drive_link",
      },
    ],
  },
  "Starlink Tracking": {
    showOnProjects: false,
    status: "archived",
    repoTitle: "Satellite-Tracker",
    repoUrl: "https://github.com/Criss-Wang/Satellite-Tracker",
    stars: 0,
    pushedAt: "2022-12-01T02:40:18Z",
    featuredTags: ["d3.js", "javascript", "react"],
    links: [{ label: "code", href: "https://github.com/Criss-Wang/Satellite-Tracker" }],
  },
  "Travel Planner": {
    showOnProjects: false,
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
