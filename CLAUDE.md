# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two related tools for set-theoretic geometry visualizations, both in Russian:

1. `difference_sets.py` — single-scene Manim video ("Разности точек контура") visualizing the difference set `D = {p_i − p_j}` of points sampled from a contour — circle → disk of radius 2R, square → square `[-a,a]²`, equilateral triangle → regular hexagon. Output lands in `media/videos/difference_sets/`.

2. `transform_studio.py` — interactive matplotlib GUI laboratory for the same math: arbitrary vector shapes (SVG import or presets) deformed by dragging vertices, with live-computed set operations — difference set, Minkowski dilation/erosion, opening/closing, convex hull — and a scrubbable construction animation (pairwise p−q arrows → dot cloud → fill reveal).

## Commands

```bash
# Manim video (Manim 0.22.1, Python 3.13)
.venv/bin/python -m manim -qm difference_sets.py DifferenceSets
# Quality: -ql/-qm/-qh/-qk; partial: --from_animation_number 50

# Transform Studio (needs X display / DISPLAY set)
.venv/bin/python transform_studio.py                 # окно
.venv/bin/python transform_studio.py --preset star   # стартовая фигура
.venv/bin/python transform_studio.py --svg shape.svg

# Verification (headless-safe)
.venv/bin/python transform_studio.py --selfcheck     # math checks vs analytic values
.venv/bin/python transform_studio.py --smoke         # renders all 30 preset×op combos
```

There are no test frameworks, linters, or build steps. `--selfcheck` and `--smoke` are the verification mechanism; run both after touching geometry code.

## Architecture

### difference_sets.py

- `SAMPLES` — number of sample points per contour (36).
- `circle_points` / `polygon_samples` — sample a contour (`polygon_samples` by arc length, not per-edge).
- `unique_differences` — pairwise difference vectors, deduplicated by 6-decimal rounding, sorted by (norm, angle) so the fill animation grows radially outward.
- `DifferenceSets.construct` — three contour→target pairs, each delegated to `show_difference_set` (shared choreography: title → sample dots → per-pair p_i−p_j arrows translated to origin O → batched dot fade-in → target reveal).
- `LEFT`/`RIGHT` constants fix the two-panel layout.

### transform_studio.py

Same two-panel convention (`LEFT` = source contour, `RIGHT` = result at origin O). Structure:

- **Geometry core (pure functions, no UI):** `rings_to_geom`/`geom_rings` (even-odd ring nesting → polygons with holes), `convex_sum` (exact convex Minkowski sum via angle-merge; terminates when the edge-walk returns to the start vertex pair), `cdt_triangles` (shapely constrained Delaunay), `minkowski_polygon` (general A⊕B = union of pairwise convex sums of triangles), `difference_set` (A ⊕ −A), `dilate`/`erode` (shapely buffer; square element uses mitre joins = exact), `compute_op` (dispatch).
- **Two-tier honesty:** polygon ops are exact (convex_sum/CDT); round buffers approximate arcs at quad_segs=96. `--selfcheck` asserts exact routes against analytic values.
- **SVG import:** hand-rolled parser (`parse_svg_file`/`parse_path_d`) — path d (all commands incl. S/T implicit-control reflection, elliptical arcs via endpoint→center parametrization), rect/circle/ellipse/polygon, nested `transform` matrices. No external SVG dependency. Beziers flatten at import; shapes simplify to ≤64 vertices (`simplify_adaptive`) and normalize to the left panel (`normalize_geom`).
- **UI (`TransformStudio`):** matplotlib widgets (radio groups, sliders, check buttons), custom timer-driven animation timeline (`_build_timeline`/`draw_frame`) with two choreographies — "diff" (pairwise arrows → dots → fill) and "sweep" (structuring element ghost-trails along contour) — plus vertex dragging (press/motion/release) with `light=True` fast recompute during drag.
- `_guard` flag prevents widget callbacks from re-triggering during programmatic updates.

Gotchas learned: matplotlib 3.11 removed `RadioButtons.circles` (use `ax.patches`) and `ArtistList` has no `.clear()` (remove artists individually). `convex_sum` for shapes with all-parallel edge pairs (square ⊕ −square) must stop on return-to-start, not after m+k steps, or it traces the polygon twice.

## Notes

- Text and labels are in Russian; scene content is math-education material.
- **Recreating the venv:** `manimpango` has no prebuilt wheel here and fails to compile unless `pangocairo.pc` is findable. The system lacks the dev package, but the conda env `~/miniforge3/envs/bear_vf_env` has pango/cairo. Recreate with:
  ```bash
  python3 -m venv .venv
  PKG_CONFIG_PATH=~/miniforge3/envs/bear_vf_env/lib/pkgconfig .venv/bin/pip install manim numpy
  ```
  At runtime manim loads pango via manimpango's compiled extension, so no extra env vars are needed for rendering.
- The `media/` directory (rendered SVGs, MP4s, partial movie files) is generated output; don't edit by hand.
- `.venv_stale/` is the old venv copied from another machine (broken shebangs); safe to delete.
