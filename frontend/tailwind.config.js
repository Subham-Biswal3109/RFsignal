/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Map CSS custom properties to Tailwind token names.
        available: "rgb(var(--color-available) / <alpha-value>)",
        occupied:  "rgb(var(--color-occupied)  / <alpha-value>)",
        warning:   "rgb(var(--color-warning)   / <alpha-value>)",
        // Neutral surface tokens used by Panel/components.
        surface:       "hsl(var(--surface))",
        "surface-raised": "hsl(var(--surface-raised))",
        background:    "hsl(var(--background))",
        foreground:    "hsl(var(--foreground))",
        border:        "hsl(var(--border))",
        "muted-foreground": "hsl(var(--muted-foreground))",
        destructive:   "hsl(var(--destructive))",
      },
    },
  },
  plugins: [],
};
