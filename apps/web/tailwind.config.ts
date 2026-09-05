import type { Config } from "tailwindcss";

// Shared design tokens (see app/globals.css for the CSS custom properties) —
// kept identical across projects so every dashboard uses the same palette.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--color-bg)",
        surface: "var(--color-surface)",
        "surface-elevated": "var(--color-surface-elevated)",
        "surface-raised": "var(--color-surface-raised)",
        fg: "var(--color-fg)",
        muted: "var(--color-muted)",
        subtle: "var(--color-subtle)",
        border: "var(--color-border)",
        "border-strong": "var(--color-border-strong)",
        accent: "var(--color-accent)",
        "accent-fill": "var(--color-accent-fill)",
        "accent-hover": "var(--color-accent-hover)",
        "accent-soft": "var(--color-accent-soft)",
        secondary: "var(--color-secondary)",
        "secondary-hover": "var(--color-secondary-hover)",
        gold: "var(--color-gold)",
        "gold-hover": "var(--color-gold-hover)",
        success: "var(--color-success)",
        warning: "var(--color-warning)",
        danger: "var(--color-danger)",
        "danger-soft": "var(--color-danger-soft)",
        info: "var(--color-info)",
        focus: "var(--color-focus)",
        "primary-soft": "var(--color-primary-soft)",
      },
    },
  },
  plugins: [],
};

export default config;
