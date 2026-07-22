# Task: install the "visual-explainer" personal skill

Create a personal (global) Claude Code skill so it's available across all my projects.
It captures my house style for HTML learning-reports and concept explainers.

## Steps

1. Create the directory `~/.claude/skills/visual-explainer/` (make parent dirs as needed).
2. Write the SKILL.md below into `~/.claude/skills/visual-explainer/SKILL.md`.

````bash
mkdir -p ~/.claude/skills/visual-explainer
cat > ~/.claude/skills/visual-explainer/SKILL.md << 'SKILL_EOF'
---
name: visual-explainer
description: >-
  Build single-file HTML explainers and learning reports that teach a concept or
  document a work session through semantic-color diagrams, hand-built inline SVG,
  and a two-lens (business / under-the-hood) narration. Use this skill WHENEVER the
  user asks for a report, session recap, concept explainer, pipeline walkthrough,
  architecture write-up, "explica isso visualmente", "faz um HTML com diagramas",
  or any deliverable meant to make an idea or a learning as visual as possible —
  even if they don't say the word "skill" or "diagram". Default to this skill for
  study/session reports and for explaining how a system, pipeline, or data
  transformation works.
---

# Visual Explainer

A house style for turning a concept, pipeline, or work session into a single self-contained
HTML page where **the diagrams carry the meaning**, not the prose. Optimized for teaching and
documentation, in Brazilian Portuguese by default (mix English technical terms freely).

The north-star reference is `insurance-copilot-pipeline-explicada.html`: a five-stage pipeline
where the data changes shape at each stage and each SVG draws that shape.

## The one rule that matters most

**Color is semantic, never decorative.** Before building, fix 2–3 accent colors and assign each
a *meaning* that holds across the entire page — the flow strip, the cards, the SVG internals, the
legends. When the reader sees the color, they should already know what it means before reading.
This is what makes the page stop feeling like decoration and start feeling like a legend.

Never introduce a color that doesn't mean something. If you have 4+ accents, you've lost the plot.

## Design tokens (the reference palette)

```css
:root{
  /* surfaces */
  --bg:#faf9f7; --card:#fff; --ink:#1f1f1f; --muted:#6b6b6b; --faint:#a5a19b;
  --line:#e3e0db; --line2:#d0ccc6;
  /* SEMANTIC ACCENTS — reassign meanings per topic, keep to 2–3 */
  --now:#0f766e; --now-bg:#e6f4f1;   /* teal  = code / deterministic / no-AI / anchor / PK */
  --pk:#7a5af5;  --pk-bg:#efeafe;    /* purple = AI / model / primary key */
  --fk:#c2410c;  --fk-bg:#fbeadf;    /* orange = derived / foreign key / the "moving part" */
}
```

The *hues* are the house identity; the *meanings* are per-topic. For a LangGraph piece the
meanings might be "graph layer / runtime layer / the bridge"; for a pipeline "code / AI / derived".
Reassign freely, but write the meaning in a comment and hold it everywhere.

## Typography with roles

- **Sans** (`-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`) → all human prose.
- **Monospace** (`ui-monospace, Menlo, monospace`) → everything machine-flavored: chips, tags,
  ids, `entra:/sai:`, code tokens, table/field names.

The reader should never confuse an explanation with a technical token. The typeface does that work.

## The structural patterns (use as many as fit)

1. **Thesis lead.** Open with ONE memorable sentence stating the single idea, with the pivot
   bolded. If you can't compress the page to one sentence, you don't know the page yet.

2. **Flow strip.** A horizontal overview of the whole thing at the top: steps separated by arrows,
   and the arrows are *labeled with the kind of transition* (código / IA / merge). Orient before
   detailing. Collapses to stacked on mobile.

3. **Stage / concept cards.** Each unit = a numbered circle + title + a semantic tag (which accent
   it belongs to) + a one-line subtitle + a two-column body: `viz | text`. Grid
   `grid-template-columns: 270px 1fr`, collapsing to one column under ~640px.

4. **Hand-built inline SVG that shows the SHAPE of the thing.** Not a stock icon — draw the actual
   structure at that step (a document, a nested tree, rows with PK/FK badges, an ER graph, two
   stacked layer-bands). viewBox around `0 0 260 150`. Every SVG gets a plain-language caption at
   the bottom, and a small legend when it uses encoded colors/badges.

5. **The two-lens narration.** Explain every unit twice, in two labeled blocks:
   - **"Para o negócio"** — why it matters, in plain terms, no jargon.
   - **"Sob o capô"** — how it actually works, technical.
   This lets one page serve both a stakeholder and an engineer without choosing.

6. **I/O chips.** Make each transformation explicit with monospace chips: `entra: X` / `sai: Y`.
   Bold the payload. These turn "a process happened" into "this became that".

7. **Payoff box.** Close with a concrete question the whole construction answers, in the anchor
   accent. Prove the effort was worth it.

## Diagram vocabulary — pick the shape that fits the idea

Don't default to boxes-and-arrows for everything. Match the diagram to what's being taught:

| Idea being shown | Diagram to reach for |
|---|---|
| Linear sequence of transformations | **Flow strip** (steps + labeled arrows) |
| Nested / hierarchical structure | **Tree** (parent → children, indented nodes) |
| How records link / relational data | **Row diagram with PK/FK badges** (colored id chips) |
| Whole schema at a glance | **ER mini-diagram** (tables + FK lines following ids) |
| "Which layer owns what" / separation of concerns | **Stacked layer-bands** (top band vs bottom band, a boundary line the data can't cross, one vertical channel that bridges) |
| Lifecycle / status transitions | **State machine** (nodes + directed labeled edges, a start and an END) |
| Contrasting two approaches | **Comparison table** (2 cols) or **side-by-side before/after** |
| Routing / branching / a decision | **Decision diagram** (one node fans out to conditional paths) |
| A quantity accumulating or overwriting | **Before/after strip** showing the field's value at each step |
| Proportions / counts | Simple inline **bar** built from `<rect>`s (avoid chart libs for a handful of values) |

When a table communicates faster than a picture, use the table. "Most visual" ≠ "most SVG" — it
means "fastest to understand". A tight comparison table often beats a diagram.

## Layout scaffold

Single file. Inline `<style>` in `:root`, no external deps, no chart libraries, no JS unless an
interaction genuinely helps. `.wrap{max-width:880px;margin:0 auto}`. Cards are white on the warm
`--bg` paper, `border:1px solid var(--line)`, `border-radius:~11px`. Section headers are small,
uppercase, letter-spaced, muted. Body text 15px/1.6.

## Build checklist (run before delivering)

- [ ] Can I state the whole page in one bolded sentence? (thesis lead)
- [ ] Do I have exactly 2–3 accent colors, each with a written meaning held everywhere?
- [ ] Does each SVG draw the *shape of the thing*, with a caption + legend where needed?
- [ ] Is every unit explained in both lenses (business / under-the-hood)?
- [ ] Are transformations made explicit with I/O chips?
- [ ] Monospace for machine tokens, sans for prose — no mixing?
- [ ] Does it close on a concrete payoff?
- [ ] Responsive: flow strip and 2-col bodies collapse cleanly on mobile?
- [ ] One accent per meaning — did I remove any decorative color that means nothing?

## What to avoid

- Generic AI-report look: cream + serif + terracotta #D97757, or black + acid-green. Those read
  as defaults. This house style is warm-paper + semantic teal/purple/orange with a monospace/sans split.
- Colors that don't mean anything. Decoration for its own sake.
- Stock icons where a structural drawing would teach more.
- Walls of prose. If a paragraph can become a diagram + caption, do that.
- Chart libraries for 3–5 numbers. Draw rects.
SKILL_EOF
````

3. Confirm it landed: `ls -la ~/.claude/skills/visual-explainer/ && head -20 ~/.claude/skills/visual-explainer/SKILL.md`

## Note
If `~/.claude/skills/` did not already exist before this Claude Code session started,
restart Claude Code once after creating it so the new directory gets watched. Editing the
SKILL.md later is picked up live without restart.
