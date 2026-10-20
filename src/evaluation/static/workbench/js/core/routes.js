/* Single source of truth for navigation. Every entry maps to a real page module. */
export const ROUTES = [
  { id: "overview",   path: "/overview",   label: "Overview",                title: "Overview",                group: "Workspace",  icon: "overview",   load: () => import("../pages/overview.js") },
  { id: "dataset",    path: "/dataset",    label: "Dataset & Privacy",       title: "Dataset & Privacy",       group: "Pipeline",   icon: "dataset",    load: () => import("../pages/dataset.js") },
  { id: "training",   path: "/training",   label: "Training",                title: "Training",                group: "Pipeline",   icon: "training",   load: () => import("../pages/training.js") },
  { id: "screening",  path: "/screening",  label: "Adapter Screening",       title: "Adapter Screening",       group: "Pipeline",   icon: "screening",  load: () => import("../pages/screening.js") },
  { id: "packaging",  path: "/packaging",  label: "Secure Packaging",        title: "Secure Packaging",        group: "Pipeline",   icon: "packaging",  load: () => import("../pages/packaging.js") },
  { id: "deployment", path: "/deployment", label: "Deployment Verification", title: "Deployment Verification", group: "Pipeline",   icon: "deployment", load: () => import("../pages/deployment.js") },
  { id: "evaluation", path: "/evaluation", label: "Evaluation & Reports",    title: "Evaluation & Reports",    group: "Insights",   icon: "evaluation", load: () => import("../pages/evaluation.js") },
  { id: "history",    path: "/history",    label: "Run History",             title: "Run History",             group: "Insights",   icon: "history",    load: () => import("../pages/history.js") },
  { id: "settings",   path: "/settings",   label: "Settings",                title: "Settings",                group: "System",     icon: "settings",   load: () => import("../pages/settings.js") },
];

export const DEFAULT_PATH = "/overview";
export const routeForPath = (path) => ROUTES.find((r) => r.path === path) || null;
