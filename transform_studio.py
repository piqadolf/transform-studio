"""Transform Studio — интерактивная лаборатория теоретико-множественной геометрии.

Продолжение идеи difference_sets.py: вместо готового видео — живой стенд,
где произвольную векторную фигуру можно деформировать мышью и смотреть,
что с ней делают операции над множествами точек.

Операции
--------
  Разностное множество   D = {p − q : p, q ∈ A}  (= A ⊕ (−A))
  Расширение (dilation)  A ⊕ B
  Эрозия (erosion)       A ⊖ B
  Открытие / Закрытие    (A ⊖ B) ⊕ B  /  (A ⊕ B) ⊖ B
  Выпуклая оболочка      conv(A)

B — структурирующий элемент: круг (радиус) или квадрат (сторона).

Честность вычислений (два уровня, как договорились)
---------------------------------------------------
* Точно: все операции над многоугольниками — точная многоугольная
  арифметика. Разностное множество считается через триангуляцию CDT и
  попарные суммы Минковского выпуклых кусков (объединение — точное).
  Расширение/эрозия квадратом — точные (mitre-буфер = сумма с квадратом).
* С высокой точностью: дуги (расширение кругом) аппроксимируются
  96-сегментными ломаными; кривые Безье из SVG спрямляются при импорте
  и упрощаются до ≤64 вершин для интерактивности.

Запуск
------
  .venv/bin/python transform_studio.py                 # окно
  .venv/bin/python transform_studio.py --preset star
  .venv/bin/python transform_studio.py --svg shape.svg
  .venv/bin/python transform_studio.py --selfcheck     # проверка математики
  .venv/bin/python transform_studio.py --smoke         # headless-рендер всех режимов

Управление: тащить вершины контура мышью; кнопки/слайдеры внизу;
«Играть» — анимация построения (пары p−q → облако разностей → заливка).
"""

import math
import os
import re
import sys
import time
import argparse
import warnings

import numpy as np

import matplotlib
_headless = sys.platform.startswith("linux") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)
if "--smoke" in sys.argv or _headless:
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Patch, PathPatch, Circle, Rectangle, FancyArrow
from matplotlib.widgets import RadioButtons, Slider, Button, CheckButtons

import shapely
from shapely.geometry import Polygon, Point
from shapely import make_valid, union_all
from shapely import affinity

warnings.filterwarnings("ignore", category=UserWarning, module="shapely")

# ----------------------------------------------------------------------------
# Константы раскладки (в координатах данных; одна общая система осей)
# ----------------------------------------------------------------------------
LEFT = np.array([-2.15, 0.0])    # центр левой панели (исходный контур)
RIGHT = np.array([2.15, 0.0])    # центр правой панели (результат, точка O)
SHAPE_R = 1.2                    # радиус нормализации фигуры
XLIM = (-4.6, 4.6)
YLIM = (-3.05, 3.05)

OPS = ["diff", "dilate", "erode", "open", "close", "hull"]
OP_LABELS = {
    "diff": "Разности D={p−q}",
    "dilate": "Расширение A⊕B",
    "erode": "Эрозия A⊖B",
    "open": "Открытие",
    "close": "Закрытие",
    "hull": "Выпуклая оболочка",
}
OP_TITLES = {
    "diff": "D = {p−q | p,q ∈ A}",
    "dilate": "A ⊕ B = {a+b}",
    "erode": "A ⊖ B = {x : x+B ⊆ A}",
    "open": "(A ⊖ B) ⊕ B",
    "close": "(A ⊕ B) ⊖ B",
    "hull": "conv(A)",
}
OP_COLORS = {
    "diff": "#3b76c4", "dilate": "#3f9d5a", "erode": "#e08214",
    "open": "#8064a7", "close": "#2aa198", "hull": "#777777",
}
ELEMENTS = {"disk": "Круг", "square": "Квадрат"}

# ----------------------------------------------------------------------------
# Геометрия: общие помощники
# ----------------------------------------------------------------------------

def iter_polygons(geom):
    """Все полигоны из геометрии (Polygon/MultiPolygon/Collection)."""
    if geom is None or geom.is_empty:
        return []
    t = geom.geom_type
    if t == "Polygon":
        return [geom]
    if t in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(iter_polygons(g))
        return out
    return []


def polygonal(geom):
    """Оставить только полигональную часть (make_valid может дать линии)."""
    parts = iter_polygons(geom)
    if not parts:
        return Polygon()
    return union_all(parts) if len(parts) > 1 else parts[0]


def ring_points(ring):
    """Координаты замкнутого кольца без последней (дублирующей) точки."""
    pts = np.asarray(ring.coords, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def orient_ccw(pts):
    """Ориентировать замкнутый контур против часовой стрелки."""
    x, y = pts[:, 0], pts[:, 1]
    area2 = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1])
    return pts if area2 > 0 else pts[::-1]


def rings_to_geom(rings):
    """Кольца (list[np.array]) -> валидная (Multi)Polygon с дырками по чётности."""
    rings = [np.asarray(r, dtype=float) for r in rings if len(r) >= 3]
    if not rings:
        return Polygon()
    raw = []
    for r in rings:
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[:1]])
        try:
            p = make_valid(Polygon(r))
        except Exception:
            continue
        raw.extend(iter_polygons(p))
    if not raw:
        return Polygon()
    # чётно-нечётная вложенность: чётная глубина — внешний контур
    reps = [p.representative_point() for p in raw]
    shells, holes = [], {}
    for i, p in enumerate(raw):
        containing = [q for j, q in enumerate(raw) if j != i and q.contains(reps[i])]
        depth = len(containing)
        if depth % 2 == 0:
            shells.append(p)
        else:
            parent = min(containing, key=lambda g: g.area)
            holes.setdefault(id(parent), []).append(p)
    if not shells:
        shells = [max(raw, key=lambda g: g.area)]
    built = []
    for s in shells:
        hs = [ring_points(h.exterior) for h in holes.get(id(s), [])]
        built.append(Polygon(ring_points(s.exterior), holes=hs))
    return polygonal(union_all(built))


def geom_rings(geom):
    """Геометрия -> список колец (внешние + дырки) для редактирования."""
    rings = []
    for p in iter_polygons(geom):
        rings.append(ring_points(p.exterior))
        for h in p.interiors:
            rings.append(ring_points(h))
    return [r for r in rings if len(r) >= 3]


def simplify_adaptive(geom, max_vertices=64, rel_tol=0.004):
    """Упрощать, пока суммарное число вершин не станет <= max_vertices."""
    diag = max(geom.bounds[2] - geom.bounds[0], geom.bounds[3] - geom.bounds[1], 1e-9)
    tol = rel_tol * diag
    for _ in range(30):
        n = sum(len(r) for r in geom_rings(geom))
        if n <= max_vertices or tol > 2 * diag:
            return geom
        tol *= 1.6
        geom = geom.simplify(tol, preserve_topology=True)
    return geom


def normalize_geom(geom, center, target_r):
    """Масштабировать и центрировать геометрию."""
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max((maxx - minx) / 2, (maxy - miny) / 2, 1e-9)
    s = target_r / half
    geom = affinity.translate(geom, -cx, -cy)
    geom = affinity.scale(geom, s, s, origin=(0, 0))
    return affinity.translate(geom, center[0], center[1])


def sample_contour(geom, n):
    """n точек, равномерно по длине внешнего контура крупнейшего полигона."""
    polys = iter_polygons(geom)
    if not polys:
        return []
    pts = ring_points(max(polys, key=lambda p: p.area).exterior)
    if len(pts) < 2:
        return []
    closed = np.vstack([pts, pts[:1]])
    seg = np.diff(closed, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    total = lens.sum()
    if total <= 0:
        return []
    targets = np.arange(n) * total / n
    idx = np.searchsorted(np.cumsum(lens), targets, side="right").clip(0, len(seg) - 1)
    passed = np.concatenate([[0.0], np.cumsum(lens)[:-1]])[idx]
    ratio = ((targets - passed) / lens[idx]).clip(0, 1)
    return closed[idx] + ratio[:, None] * seg[idx]


def unique_differences(points, cap=1600):
    """Все попарные разности без повторов, по возрастанию |v| (как в видео)."""
    if len(points) < 2:
        return []
    P = np.asarray(points)
    V = (P[:, None, :] - P[None, :, :]).reshape(-1, 2)
    keys = np.round(V, 5)
    _, keep = np.unique(keys, axis=0, return_index=True)
    V = V[keep]
    if len(V) > cap:
        V = V[np.random.default_rng(0).choice(len(V), cap, replace=False)]
    norm = np.linalg.norm(V, axis=1)
    ang = np.arctan2(V[:, 1], V[:, 0])
    return V[np.lexsort((ang, norm))]

# ----------------------------------------------------------------------------
# Сумма Минковского выпуклых многоугольников (слияние рёбер по углу)
# ----------------------------------------------------------------------------

def convex_sum(P, Q):
    """Точная сумма Минковского двух выпуклых CCW-многоугольников."""
    P = orient_ccw(np.asarray(P, dtype=float))
    Q = orient_ccw(np.asarray(Q, dtype=float))
    m, k = len(P), len(Q)
    i = int(np.lexsort((P[:, 0], P[:, 1]))[0])
    j = int(np.lexsort((Q[:, 0], Q[:, 1]))[0])
    i0, j0 = i, j
    out = []
    while True:
        out.append(P[i % m] + Q[j % k])
        eP = P[(i + 1) % m] - P[i % m]
        eQ = Q[(j + 1) % k] - Q[j % k]
        cross = eP[0] * eQ[1] - eP[1] * eQ[0]
        if cross > 0:
            i += 1
        elif cross < 0:
            j += 1
        else:
            i += 1
            j += 1
        if i % m == i0 and j % k == j0:
            break
        if len(out) > m + k:  # защита от бесконечного цикла
            return None
    O = np.round(np.asarray(out), 9)
    keep = np.ones(len(O), bool)
    keep[1:] = np.linalg.norm(np.diff(O, axis=0), axis=1) > 1e-9
    if len(O) > 1 and np.linalg.norm(O[0] - O[-1]) <= 1e-9:
        keep[-1] = False
    O = O[keep]
    return O if len(O) >= 3 else None


def cdt_triangles(geom):
    """Триангуляция (constrained Delaunay) полигональной части геометрии."""
    tris = []
    for p in iter_polygons(geom):
        try:
            pieces = shapely.constrained_delaunay_triangles(p)
        except Exception:
            continue
        for t in getattr(pieces, "geoms", [pieces]):
            if t.geom_type == "Polygon" and t.area > 1e-12:
                tris.append(ring_points(t.exterior))
    return tris


def _hull_vertices(geom):
    """Если геометрия уже выпукла — вернуть её вершины, иначе None."""
    hull = geom.convex_hull
    if hull.geom_type != "Polygon" or hull.is_empty:
        return None
    if abs(hull.area - geom.area) <= 1e-9 * max(geom.area, 1.0):
        return ring_points(hull.exterior)
    return None


def minkowski_polygon(A, B):
    """A ⊕ B для произвольных многоугольников: CDT + попарные выпуклые суммы."""
    ha = _hull_vertices(A)
    hb = _hull_vertices(B)
    if ha is not None and hb is not None:
        s = convex_sum(ha, hb)
        return Polygon(s) if s is not None else Polygon()
    ta = cdt_triangles(A) or ([ha] if ha is not None else [])
    tb = cdt_triangles(B) or ([hb] if hb is not None else [])
    if not ta or not tb:
        return Polygon()
    parts = []
    for pa in ta:
        for pb in tb:
            s = convex_sum(pa, pb)
            if s is not None:
                parts.append(Polygon(s))
    return union_all(parts) if parts else Polygon()


def difference_set(geom, light=False):
    """D = A ⊕ (−A). light=True — упрощённая фигура (быстрый пересчёт при драге)."""
    if geom.is_empty:
        return Polygon()
    work = geom
    if light:
        diag = math.dist(geom.bounds[:2], geom.bounds[2:])
        work = geom.simplify(0.035 * max(diag, 1e-9), preserve_topology=True)
        if work.is_empty:
            work = geom
    mirrored = affinity.scale(work, -1.0, -1.0, origin=(0, 0))
    return minkowski_polygon(work, mirrored)

# ----------------------------------------------------------------------------
# Операции со структурирующим элементом
# ----------------------------------------------------------------------------

def dilate(geom, element, size):
    if element == "disk":
        return polygonal(geom.buffer(size, quad_segs=96, join_style="round"))
    half = size / 2.0
    return polygonal(geom.buffer(half, quad_segs=96, join_style="mitre", mitre_limit=1e9))


def erode(geom, element, size):
    if geom.is_empty:
        return geom
    if element == "disk":
        d, join = size, "round"
    else:
        d, join = size / 2.0, "mitre"
    return polygonal(geom.buffer(-d, quad_segs=96, join_style=join, mitre_limit=1e9))


def compute_op(op, geom, element, size, light=False):
    if geom.is_empty:
        return Polygon()
    if op == "diff":
        return difference_set(geom, light=light)
    if op == "dilate":
        return dilate(geom, element, size)
    if op == "erode":
        return erode(geom, element, size)
    if op == "open":
        return dilate(erode(geom, element, size), element, size)
    if op == "close":
        return erode(dilate(geom, element, size), element, size)
    if op == "hull":
        return polygonal(geom.convex_hull)
    raise ValueError(op)

# ----------------------------------------------------------------------------
# SVG-импорт (свой парсер: path d, базовые фигуры, transform)
# ----------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_TOK = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(" + _NUM + r")")
_TF = re.compile(r"(\w+)\s*\(([^)]*)\)")


def _mat_mul(m1, m2):
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _apply(m, pt):
    a, b, c, d, e, f = m
    x, y = pt
    return (a * x + c * y + e, b * x + d * y + f)


def parse_transform(text):
    """transform="..." -> матрица 6."""
    m = (1, 0, 0, 1, 0, 0)
    if not text:
        return m
    for name, args_s in _TF.findall(text):
        args = [float(v) for v in re.split(r"[\s,]+", args_s.strip()) if v]
        if name == "translate" and args:
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            m = _mat_mul((1, 0, 0, 1, tx, ty), m)
        elif name == "scale" and args:
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            m = _mat_mul((sx, 0, 0, sy, 0, 0), m)
        elif name == "rotate" and args:
            ang = math.radians(args[0])
            ca, sa = math.cos(ang), math.sin(ang)
            r = (ca, sa, -sa, ca, 0, 0)
            if len(args) == 3:
                cx, cy = args[1], args[2]
                r = _mat_mul(_mat_mul((1, 0, 0, 1, cx, cy), r), (1, 0, 0, 1, -cx, -cy))
            m = _mat_mul(r, m)
        elif name == "matrix" and len(args) == 6:
            m = _mat_mul(tuple(args), m)
    return m


def _flat_cubic(p0, p1, p2, p3):
    chord = math.dist(p0, p1) + math.dist(p1, p2) + math.dist(p2, p3)
    n = int(np.clip(chord * 6, 8, 40))
    t = np.linspace(0, 1, n + 1)[1:]
    mt = 1 - t
    x = (mt ** 3 * p0[0] + 3 * mt ** 2 * t * p1[0] + 3 * mt * t ** 2 * p2[0] + t ** 3 * p3[0])
    y = (mt ** 3 * p0[1] + 3 * mt ** 2 * t * p1[1] + 3 * mt * t ** 2 * p2[1] + t ** 3 * p3[1])
    return list(zip(x, y))


def _flat_quad(p0, p1, p2):
    chord = math.dist(p0, p1) + math.dist(p1, p2)
    n = int(np.clip(chord * 8, 8, 36))
    t = np.linspace(0, 1, n + 1)[1:]
    mt = 1 - t
    x = mt ** 2 * p0[0] + 2 * mt * t * p1[0] + t ** 2 * p2[0]
    y = mt ** 2 * p0[1] + 2 * mt * t * p1[1] + t ** 2 * p2[1]
    return list(zip(x, y))


def _arc_points(p0, p1, rx, ry, phi_deg, large, sweep):
    """Эллиптическая дуга (endpoint -> center parametrization, по спецификации SVG)."""
    if p0 == p1 or rx <= 0 or ry <= 0:
        return []
    phi = math.radians(phi_deg)
    cosp, sinp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1 = cosp * dx + sinp * dy
    y1 = -sinp * dx + cosp * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
    den = rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2
    co = math.sqrt(max(num, 0) / max(den, 1e-18))
    if large == sweep:
        co = -co
    cxp = co * rx * y1 / ry
    cyp = -co * ry * x1 / rx
    cx = cosp * cxp - sinp * cyp + (p0[0] + p1[0]) / 2
    cy = sinp * cxp + cosp * cyp + (p0[1] + p1[1]) / 2

    def ang(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1, min(1, dot / max(n, 1e-18))))
        return a if ux * vy - uy * vx >= 0 else -a

    th1 = ang(1, 0, (x1 - cxp) / rx, (y1 - cyp) / ry)
    dth = ang((x1 - cxp) / rx, (y1 - cyp) / ry, (-x1 - cxp) / rx, (-y1 - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    if sweep and dth < 0:
        dth += 2 * math.pi
    n = int(np.clip(abs(dth) / (2 * math.pi) * 96, 6, 96))
    pts = []
    for i in range(1, n + 1):
        t = th1 + dth * i / n
        px = rx * math.cos(t)
        py = ry * math.sin(t)
        pts.append((cosp * px - sinp * py + cx, sinp * px + cosp * py + cy))
    return pts


def parse_path_d(d):
    """d-атрибут -> список незамкнутых ломаных (subpath'и)."""
    tokens = []
    for m in _TOK.finditer(d):
        if m.group(1):
            tokens.append(m.group(1))
        else:
            tokens.append(float(m.group(2)))
    paths, cur = [], []
    pen = (0.0, 0.0)
    start = (0.0, 0.0)
    prev_cubic_ctrl = None
    prev_quad_ctrl = None
    cmd = None
    i = 0

    def need(k):
        nonlocal i
        vals = tokens[i:i + k]
        i += k
        return vals

    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            cmd = tok
            i += 1
        else:  # неявное повторение последней команды
            if cmd == "M":
                cmd = "L"
            elif cmd == "m":
                cmd = "l"
        rel = cmd.islower()
        C = cmd.upper()
        if C == "M":
            if len(cur) > 1:
                paths.append(cur)
            x, y = need(2)
            pen = (x + pen[0], y + pen[1]) if rel else (x, y)
            start = pen
            cur = [pen]
        elif C == "L":
            x, y = need(2)
            pen = (x + pen[0], y + pen[1]) if rel else (x, y)
            cur.append(pen)
        elif C == "H":
            (x,) = need(1)
            pen = (x + pen[0], pen[1]) if rel else (x, pen[1])
            cur.append(pen)
        elif C == "V":
            (y,) = need(1)
            pen = (pen[0], y + pen[1]) if rel else (pen[0], y)
            cur.append(pen)
        elif C in ("C", "S"):
            if C == "C":
                x1, y1, x2, y2, x, y = need(6)
                c1 = (x1 + pen[0], y1 + pen[1]) if rel else (x1, y1)
                c2 = (x2 + pen[0], y2 + pen[1]) if rel else (x2, y2)
                p = (x + pen[0], y + pen[1]) if rel else (x, y)
            else:
                x2, y2, x, y = need(4)
                c1 = (2 * pen[0] - prev_cubic_ctrl[0], 2 * pen[1] - prev_cubic_ctrl[1]) \
                    if prev_cubic_ctrl else pen
                c2 = (x2 + pen[0], y2 + pen[1]) if rel else (x2, y2)
                p = (x + pen[0], y + pen[1]) if rel else (x, y)
            cur.extend(_flat_cubic(pen, c1, c2, p))
            prev_cubic_ctrl = c2
            pen = p
        elif C in ("Q", "T"):
            if C == "Q":
                x1, y1, x, y = need(4)
                c = (x1 + pen[0], y1 + pen[1]) if rel else (x1, y1)
                p = (x + pen[0], y + pen[1]) if rel else (x, y)
            else:
                x, y = need(2)
                c = (2 * pen[0] - prev_quad_ctrl[0], 2 * pen[1] - prev_quad_ctrl[1]) \
                    if prev_quad_ctrl else pen
                p = (x + pen[0], y + pen[1]) if rel else (x, y)
            cur.extend(_flat_quad(pen, c, p))
            prev_quad_ctrl = c
            pen = p
        elif C == "A":
            rx, ry, rot, laf, sf, x, y = need(7)
            p = (x + pen[0], y + pen[1]) if rel else (x, y)
            cur.extend(_arc_points(pen, p, rx, ry, rot, int(laf) != 0, int(sf) != 0))
            pen = p
        elif C == "Z":
            if len(cur) > 1:
                paths.append(cur)
            cur = []
            pen = start
    if len(cur) > 1:
        paths.append(cur)
    return paths


def parse_svg_file(fname):
    """SVG-файл -> (список колец в координатах SVG, предупреждения)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(fname)
    root = tree.getroot()
    warns = []
    polys = []  # (mat, точки)

    def local(tag):
        return tag.rsplit("}", 1)[-1].lower()

    def walk(node, mat):
        tf = node.get("transform")
        m = _mat_mul(parse_transform(tf), mat) if tf else mat
        tag = local(node.tag)
        try:
            if tag == "path" and node.get("d"):
                for sub in parse_path_d(node.get("d")):
                    polys.append((m, sub))
            elif tag == "rect":
                x, y = float(node.get("x", 0)), float(node.get("y", 0))
                w, h = float(node.get("width", 0)), float(node.get("height", 0))
                if w > 0 and h > 0:
                    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    polys.append((m, pts))
            elif tag == "circle":
                cx, cy, r = (float(node.get(k, 0)) for k in ("cx", "cy", "r"))
                if r > 0:
                    t = np.linspace(0, 2 * np.pi, 129)[:-1]
                    polys.append((m, [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t]))
            elif tag == "ellipse":
                cx, cy = float(node.get("cx", 0)), float(node.get("cy", 0))
                rx, ry = float(node.get("rx", 0)), float(node.get("ry", 0))
                if rx > 0 and ry > 0:
                    t = np.linspace(0, 2 * np.pi, 129)[:-1]
                    polys.append((m, [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]))
            elif tag in ("polygon", "polyline"):
                vals = [float(v) for v in re.findall(_NUM, node.get("points", ""))]
                pairs = list(zip(vals[::2], vals[1::2]))
                if len(pairs) >= 3:
                    polys.append((m, pairs))
        except (ValueError, IndexError) as exc:
            warns.append(f"{tag}: пропуск ({exc})")
        for child in node:
            walk(child, m)

    walk(root, (1, 0, 0, 1, 0, 0))
    if not polys:
        return [], ["в SVG не найдено фигур"]
    rings = [np.array([_apply(m, p) for p in pts]) for m, pts in polys]
    return rings, warns

# ----------------------------------------------------------------------------
# Пресеты
# ----------------------------------------------------------------------------

def preset_rings(name):
    if name == "circle":
        t = np.linspace(0, 2 * np.pi, 65)[:-1]
        return [np.column_stack([np.cos(t), np.sin(t)])]
    if name == "square":
        s = 0.9
        return [np.array([(-s, -s), (s, -s), (s, s), (-s, s)])]
    if name == "triangle":
        r = 1.1
        t = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
        return [np.column_stack([r * np.cos(t), r * np.sin(t)])]
    if name == "star":
        n, r1, r2 = 5, 1.15, 0.5
        t = np.pi / 2 + np.pi * np.arange(2 * n) / n
        rr = np.where(np.arange(2 * n) % 2 == 0, r1, r2)
        return [np.column_stack([rr * np.cos(t), rr * np.sin(t)])]
    if name == "ell":  # L-фигура: невыпуклая, с интересной разностью
        return [np.array([(-0.9, -0.9), (0.9, -0.9), (0.9, 0.0), (0.0, 0.0),
                          (0.0, 0.9), (-0.9, 0.9)])]
    raise ValueError(name)


PRESETS = ["circle", "square", "triangle", "star", "ell"]
PRESET_LABELS = {"circle": "Круг", "square": "Квадрат", "triangle": "Треугольник",
                 "star": "Звезда", "ell": "L-фигура"}

# ----------------------------------------------------------------------------
# Отрисовка shapely -> matplotlib
# ----------------------------------------------------------------------------

def geom_to_patch(geom, **kw):
    """(Multi)Polygon с дырками -> PathPatch (None, если пусто)."""
    polys = iter_polygons(geom)
    if not polys:
        return None
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 3:
                continue
            verts.append(xy)
            codes.append([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
    if not verts:
        return None
    path = Path(np.vstack(verts), np.hstack(codes))
    return PathPatch(path, **kw)

# ----------------------------------------------------------------------------
# Самопроверка математики (запускается на старте, подробно — с --selfcheck)
# ----------------------------------------------------------------------------

def selfcheck(verbose=False):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        if verbose:
            print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
        ok = ok and cond

    # 1. Разностное множество квадрата: точный квадрат [-2a,2a]^2
    sq = Polygon(preset_rings("square")[0])
    D = difference_set(sq)
    side = 2 * 0.9
    check("разность квадрата = квадрат 2a×2a",
          abs(D.area - (2 * side) ** 2) < 1e-6, f"area={D.area:.6f}")

    # 2. Разностное множество треугольника: правильный шестиугольник
    #    равносторонний треугольник со стороной s -> шестиугольник со стороной s
    tri = Polygon(preset_rings("triangle")[0])
    Dt = difference_set(tri)
    s = np.linalg.norm(np.asarray(tri.exterior.coords)[1] -
                       np.asarray(tri.exterior.coords)[0])
    hex_exact = (3 * np.sqrt(3) / 2) * s ** 2
    check("разность треугольника = шестиугольник",
          abs(Dt.area - hex_exact) < 1e-6 * hex_exact, f"area={Dt.area:.6f} vs {hex_exact:.6f}")

    # 3. Все вершинные разности лежат в D (невыпуклая L-фигура)
    L = Polygon(preset_rings("ell")[0])
    DL = difference_set(L)
    V = np.asarray(L.exterior.coords)
    inside = all(DL.buffer(1e-9).covers(Point(*(u - w))) for u in V for w in V)
    check("вершинные разности L-фигуры ⊆ D", inside)

    # 4. Эрозия ⊆ фигура; открытие ⊆ фигура (допуск — погрешность буферизации)
    star = Polygon(preset_rings("star")[0])
    er = erode(star, "disk", 0.15)
    op = dilate(er, "disk", 0.15)
    check("эрозия ⊆ A", star.buffer(1e-6).covers(er) if not er.is_empty else True)
    check("открытие ⊆ A", op.difference(star).area < 1e-6 * max(star.area, 1.0))

    # 5. Расширение квадратом = точная сумма Минковского
    d_mitre = dilate(sq, "square", 0.6)
    B = Polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)])
    d_exact = minkowski_polygon(sq, B)
    sym = d_mitre.symmetric_difference(d_exact).area
    check("mitre-буфер = сумма с квадратом", sym < 1e-6, f"sym={sym:.2e}")

    # 6. Кольца/дырки: круг с круглой дыркой (эталон — площади тех же 64-угольников)
    outer = preset_rings("circle")[0]
    inner = 0.5 * outer
    donut = rings_to_geom([outer, inner])
    expected = Polygon(np.vstack([outer, outer[:1]])).area \
        - Polygon(np.vstack([inner, inner[:1]])).area
    check("дырка из двух колец",
          len(iter_polygons(donut)) == 1
          and len(iter_polygons(donut)[0].interiors) == 1
          and abs(donut.area - expected) < 1e-9,
          f"area={donut.area:.6f} vs {expected:.6f}")

    # 7. Парсер path d
    subs = parse_path_d("M 0 0 L 10 0 C 10 10 20 10 20 20 Q 25 25 30 20 A 5 5 0 0 1 40 20 Z")
    npts = sum(len(s) for s in subs)
    check("парсер path d", npts > 20, f"точек={npts}")

    # 8. Дуга: полная окружность из двух дуг
    arc = parse_path_d("M 0 0 A 1 1 0 0 1 2 0 A 1 1 0 0 1 0 0")
    pts = np.vstack(arc)
    check("SVG-дуги", len(pts) > 10 and abs(pts[:, 1].max() - 1) < 0.05,
          f"y_max={pts[:, 1].max():.3f}")

    return ok

# ----------------------------------------------------------------------------
# Приложение
# ----------------------------------------------------------------------------

class TransformStudio:
    PAIR_DUR = 1.15
    DOT_BATCH_DUR = 0.10
    DOT_BATCH = 48
    FILL_DUR = 0.9
    HOLD_DUR = 0.7
    SWEEP_DUR = 4.0

    def __init__(self, preset="circle", svg=None):
        self.op = "diff"
        self.element = "disk"
        self.size = 0.45
        self.n_samples = 36
        self.show_handles = True
        self.rings = []
        self.geom = Polygon()
        self.result = Polygon()
        self.revealed = True
        self.playing = False
        self.t = 0.0
        self.T = 1.0
        self.timeline = []
        self.dyn = []
        self._guard = False
        self._drag = None
        self._last_tick = None

        self._build_figure()
        if svg:
            self.load_svg(svg)
        else:
            self.load_preset(preset)
        self._connect()

    # ------------------------------------------------------------------ UI

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14.5, 8.2))
        self.fig.patch.set_facecolor("#fafafa")
        self.ax = self.fig.add_axes([0.03, 0.26, 0.94, 0.70])
        self.ax.set_xlim(*XLIM)
        self.ax.set_ylim(*YLIM)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_facecolor("#ffffff")

        def radio(ax, labels, fn, active=0):
            r = RadioButtons(ax, labels, active=active)
            r.on_clicked(fn)
            if hasattr(r, "circles"):  # matplotlib < 3.11
                for circle in r.circles:
                    circle.set_radius(0.014)
                    circle.set_edgecolor("#666")
            else:  # matplotlib >= 3.11
                for circle in ax.patches:
                    circle.set_radius(0.014)
                    circle.set_edgecolor("#666")
            return r

        def place(x, y, w, h):
            return self.fig.add_axes([x, y, w, h])

        # группа «Фигура»
        self.fig.text(0.035, 0.225, "Фигура", fontsize=9, color="#555")
        self.preset_btn = radio(place(0.035, 0.10, 0.11, 0.115),
                                [PRESET_LABELS[p] for p in PRESETS],
                                self._on_preset, active=0)
        self.load_btn = Button(place(0.035, 0.035, 0.11, 0.045), "Загрузить SVG…")
        self.load_btn.on_clicked(lambda *_: self.load_svg_dialog())
        self.load_btn.label.set_fontsize(9)

        # группа «Операция»
        self.fig.text(0.165, 0.225, "Операция", fontsize=9, color="#555")
        self.op_btn = radio(place(0.165, 0.035, 0.165, 0.185),
                            [OP_LABELS[o] for o in OPS], self._on_op, active=0)

        # группа «Элемент B»
        self.fig.text(0.345, 0.225, "Элемент B", fontsize=9, color="#555")
        self.el_btn = radio(place(0.345, 0.175, 0.09, 0.045),
                            list(ELEMENTS.values()), self._on_element, active=0)
        self.size_sl = Slider(place(0.345, 0.115, 0.09, 0.04), "Размер", 0.05, 1.2,
                              valinit=self.size, color="#3b76c4")
        self.size_sl.label.set_fontsize(8)
        self.size_sl.on_changed(self._on_size)
        self.samples_sl = Slider(place(0.345, 0.05, 0.09, 0.04), "Точки", 12, 64,
                                 valinit=self.n_samples, valstep=1, color="#3b76c4")
        self.samples_sl.label.set_fontsize(8)
        self.samples_sl.on_changed(self._on_samples)

        # кнопки
        self.play_btn = Button(place(0.465, 0.17, 0.07, 0.05), "Играть")
        self.play_btn.on_clicked(lambda *_: self.toggle_play())
        self.handles_chk = CheckButtons(place(0.465, 0.10, 0.07, 0.05),
                                        ["Вершины"], [self.show_handles])
        self.handles_chk.on_clicked(lambda *_: self._toggle_handles())

        # скраб
        self.scrub_sl = Slider(place(0.56, 0.035, 0.41, 0.04), "Анимация",
                               0.0, 1.0, valinit=1.0, color="#e08214")
        self.scrub_sl.label.set_fontsize(8)
        self.scrub_sl.on_changed(self._on_scrub)

        # статус
        self.status = self.fig.text(0.5, 0.965, "", ha="center", fontsize=11, color="#333")
        self._build_frame_decor()

        self.timer = self.fig.canvas.new_timer(interval=33)
        self.timer.add_callback(self._on_tick)

    def _build_frame_decor(self):
        self.ax.text(LEFT[0], 2.72, "Исходный контур", ha="center", fontsize=12, color="#444")
        self.ax.text(RIGHT[0], 2.72, "Результат (перенесено в O)", ha="center",
                     fontsize=12, color="#444")
        self.ax.plot([0, 0], [YLIM[0] + 0.1, 2.45], color="#dddddd", lw=1.2, zorder=0)
        self.ax.add_patch(Circle(RIGHT, 0.035, color="#333", zorder=3))
        self.ax.annotate("O", RIGHT + np.array([0.07, -0.12]), fontsize=11, color="#333")

    # ------------------------------------------------------------ загрузка

    def load_preset(self, name):
        rings = [r.copy() for r in preset_rings(name)]
        self._set_rings([LEFT + r for r in rings])

    def load_svg_dialog(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            fname = filedialog.askopenfilename(
                title="Открыть SVG", filetypes=[("SVG", "*.svg"), ("Все файлы", "*.*")])
            root.destroy()
        except Exception:
            fname = None
        if not fname:
            self._set_status("отмена загрузки")
            return
        self.load_svg(fname)

    def load_svg(self, fname):
        try:
            rings, warns = parse_svg_file(fname)
        except Exception as exc:
            self._set_status(f"ошибка SVG: {exc}")
            return
        if not rings:
            self._set_status("SVG: фигуры не найдены")
            return
        geom = rings_to_geom(rings)
        if geom.is_empty:
            self._set_status("SVG: не удалось построить фигуру")
            return
        geom = simplify_adaptive(geom, max_vertices=64)
        geom = normalize_geom(geom, LEFT, SHAPE_R)
        self._set_rings(geom_rings(geom))
        msg = "SVG загружен"
        if warns:
            msg += f" ({len(warns)} предупреждений)"
        self._set_status(msg)

    def _set_rings(self, rings):
        self.rings = [np.asarray(r, dtype=float) for r in rings]
        self.geom = rings_to_geom(self.rings)
        self.recompute()

    # ---------------------------------------------------------- пересчёт

    def recompute(self, light=False):
        self.result = compute_op(self.op, self.geom, self.element, self.size, light=light)
        self._build_timeline()
        self.revealed = True
        if not self.playing:
            self.t = self.T
            self._guard = True
            self.scrub_sl.set_val(1.0)
            self._guard = False
        self.redraw()

    def _set_status(self, text):
        op_t = OP_TITLES[self.op]
        el = f", B = {ELEMENTS[self.element]} {self.size:.2f}" if self.op != "diff" else ""
        self.status.set_text(f"{op_t}{el}   —   {text}")

    # ------------------------------------------------------------ отрисовка

    def _clear(self):
        self.dyn = []
        for artist in list(self.ax.lines) + list(self.ax.patches) + list(self.ax.texts):
            artist.remove()
        self._build_frame_decor()

    def redraw(self):
        self._clear()

        # левая панель: контур
        for ring in self.rings:
            closed = np.vstack([ring, ring[:1]])
            self.ax.plot(closed[:, 0], closed[:, 1], color="#24425f", lw=2.2, zorder=3)
        if self.show_handles:
            for ring in self.rings:
                self.ax.plot(ring[:, 0], ring[:, 1], "o", ms=4.5, color="#c0392b",
                             mfc="white", mew=1.4, zorder=4)

        self._draw_mechanism()
        self._draw_result(alpha=0.22)
        self._set_status(self._result_note())
        self.fig.canvas.draw_idle()

    def _draw_mechanism(self):
        """Итоговый кадр слева: копии B по пути центров и контур результата поверх A."""
        pts = getattr(self, "sweep_pts", [])
        if len(pts) < 3:
            return
        for g in range(0, len(pts), max(1, len(pts) // 18)):
            self.ax.add_patch(self._element_patch(pts[g], alpha=0.16))
        self.ax.add_patch(self._element_patch(pts[0], alpha=0.8, lw=1.8))
        self.ax.plot(*pts[0], "o", ms=3, color="#3f9d5a", zorder=5)
        for poly in iter_polygons(self.result):
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.coords)
                self.ax.plot(xy[:, 0], xy[:, 1], "--", color=OP_COLORS[self.op],
                             lw=1.6, zorder=4)

    def _result_note(self):
        if self.result.is_empty:
            return "результат пуст (∅)"
        n_int = sum(len(p.interiors) for p in iter_polygons(self.result))
        note = f"площадь = {self.result.area:.3f}"
        if n_int:
            note += f", дырок: {n_int}"
        return note

    def _draw_result(self, alpha):
        if self.result.is_empty:
            self.ax.text(RIGHT[0], 0.0, "∅", ha="center", fontsize=30, color="#999", zorder=2)
            return
        geom = self._result_placed()
        patch = geom_to_patch(geom, facecolor=OP_COLORS[self.op], edgecolor=OP_COLORS[self.op],
                              alpha=alpha, lw=2.0, zorder=2)
        if patch:
            self.ax.add_patch(patch)

    def _result_placed(self):
        c = self.result.centroid
        return affinity.translate(self.result, RIGHT[0] - c.x, RIGHT[1] - c.y)

    # ------------------------------------------------------- таймлайн

    def _build_timeline(self):
        """Список (тип, данные). Всё детерминировано от t."""
        segs = []
        self.sweep_pts = []
        if self.op == "diff":
            pts = sample_contour(self.geom, self.n_samples)
            n = len(pts)
            if n >= 4:
                step = max(1, n // 12)
                pairs = [(i, (i + n // 2) % n) for i in range(0, n, step)][:14]
                dots = unique_differences(pts)
                segs = [("pair", p) for p in pairs]
                for s in range(0, len(dots), self.DOT_BATCH):
                    segs.append(("dots", dots[s:s + self.DOT_BATCH]))
                segs.append(("fill", None))
                segs.append(("hold", None))
        else:
            # центры копий B: для ⊕ — по контуру A (B снаружи касается A),
            # для ⊖ и открытия — по контуру A⊖B (B внутри касается A),
            # для закрытия — по контуру A⊕B (B снаружи касается A).
            sweep_src = {
                "dilate": self.geom,
                "close": compute_op("dilate", self.geom, self.element, self.size),
                "erode": compute_op("erode", self.geom, self.element, self.size),
                "open": compute_op("erode", self.geom, self.element, self.size),
                "hull": None,
            }[self.op]
            pts = (sample_contour(sweep_src, max(24, self.n_samples))
                   if sweep_src is not None else [])
            self.sweep_pts = pts
            if len(pts) >= 3:
                segs.append(("sweep", pts))
            segs.append(("fill", None))
            segs.append(("hold", None))
        self.timeline = segs or [("fill", None), ("hold", None)]
        self.T = sum({"pair": self.PAIR_DUR, "dots": self.DOT_BATCH_DUR,
                      "fill": self.FILL_DUR, "hold": self.HOLD_DUR,
                      "sweep": self.SWEEP_DUR}[k] for k, _ in self.timeline) or 1.0

    def _seg_durs(self):
        return {"pair": self.PAIR_DUR, "dots": self.DOT_BATCH_DUR,
                "fill": self.FILL_DUR, "hold": self.HOLD_DUR, "sull": self.SWEEP_DUR,
                "sweep": self.SWEEP_DUR}

    def _seg_at(self, t):
        durs = self._seg_durs()
        acc = 0.0
        for kind, data in self.timeline:
            dur = durs[kind]
            if t < acc + dur:
                return kind, data, (t - acc) / max(dur, 1e-9), acc
            acc += dur
        kind, data = self.timeline[-1]
        return kind, data, 1.0, self.T - durs[kind]

    # ------------------------------------------------------- анимация

    def toggle_play(self):
        if self.playing:
            self._stop()
        else:
            self._start()

    def _start(self):
        self.playing = True
        if self.t >= self.T - 1e-9:
            self.t = 0.0
        self.revealed = False
        self.play_btn.label.set_text("Пауза")
        self._last_tick = None
        self.timer.start()

    def _stop(self):
        if self.playing:
            self.playing = False
            self.play_btn.label.set_text("Играть")
            self.timer.stop()

    def _on_tick(self):
        try:
            now = time.perf_counter()
            if self._last_tick is not None:
                self.t += now - self._last_tick
            self._last_tick = now
            if self.t >= self.T:
                self.t = self.T
                self._stop()
                self.revealed = True
                self.redraw()
                self._guard = True
                self.scrub_sl.set_val(1.0)
                self._guard = False
                return
            self._guard = True
            self.scrub_sl.set_val(self.t / self.T)
            self._guard = False
            self.draw_frame(self.t)
        except Exception:
            self._stop()

    def _on_scrub(self, val):
        if self._guard:
            return
        self._stop()
        self.t = float(val) * self.T
        self.revealed = self.t >= self.T - 1e-9
        if self.revealed:
            self.redraw()
        else:
            self.draw_frame(self.t)

    def draw_frame(self, t):
        self._clear()
        kind, data, u, seg_start = self._seg_at(t)

        # статичная база: контур слева
        for ring in self.rings:
            closed = np.vstack([ring, ring[:1]])
            self.ax.plot(closed[:, 0], closed[:, 1], color="#24425f", lw=2.2, zorder=3)
        if self.show_handles:
            for ring in self.rings:
                self.ax.plot(ring[:, 0], ring[:, 1], "o", ms=4.5, color="#c0392b",
                             mfc="white", mew=1.4, zorder=4)

        if kind == "pair":
            self._draw_pair_frame(data, u, seg_start)
        elif kind == "dots":
            self._draw_dots_upto(t)
        elif kind == "fill":
            self._draw_result(alpha=0.22 * u)
            self._draw_dots_upto(t, alpha=0.5 * (1 - u))
        elif kind == "sweep":
            self._draw_sweep_frame(data, u)
        # hold: ничего поверх
        self._set_status(self._result_note())
        self.fig.canvas.draw_idle()

    def _dyn_add(self, artist):
        if isinstance(artist, Patch) and artist.axes is None:
            self.ax.add_patch(artist)
        self.dyn.append(artist)
        return artist

    def _draw_pair_frame(self, pair, u, seg_start):
        i, j = pair
        pts = sample_contour(self.geom, self.n_samples)
        a, b = pts[i], pts[j]
        v = a - b
        fade = 1.0 - max(0.0, u - 0.82) / 0.18

        # след завершённых пар — маленькие точки ±v
        acc = 0.0
        for kind, d in self.timeline:
            if acc >= seg_start - 1e-9:
                break
            if kind == "pair":
                for sgn in (1, -1):
                    pv = pts[d[0]] - pts[d[1]]
                    self._dyn_add(self.ax.plot(*(RIGHT + sgn * pv), "o", ms=3,
                                               color="#3b76c4", alpha=0.5, zorder=4)[0])
            acc += self.PAIR_DUR

        pa, pb = a, b  # контур уже в координатах левой панели
        if fade > 0:
            self._dyn_add(self.ax.plot(*pa, "o", ms=8, color="#c0392b", zorder=6,
                                       alpha=fade)[0])
            self._dyn_add(self.ax.plot(*pb, "o", ms=8, color="#b7950b", zorder=6,
                                       alpha=fade)[0])
            self._dyn_add(self.ax.annotate("$p_i$", (pa[0] + 0.06, pa[1] + 0.08),
                                           fontsize=11, color="#c0392b", alpha=fade))
            self._dyn_add(self.ax.annotate("$p_j$", (pb[0] - 0.06, pb[1] - 0.14),
                                           fontsize=11, color="#b7950b", alpha=fade))
        if u < 0.30:  # стрелка на левой панели растёт от p_j к p_i
            grow = np.clip(u / 0.30, 0, 1)
            tip = pb + grow * (pa - pb)
            self._dyn_add(FancyArrow(pb[0], pb[1], tip[0] - pb[0], tip[1] - pb[1],
                                     width=0.012, head_width=0.07, color="#b7950b",
                                     zorder=7, alpha=fade, length_includes_head=True))
        elif u < 0.55:  # перенос стрелки слева направо
            mix = (u - 0.30) / 0.25
            base = (1 - mix) * pb + mix * RIGHT
            tip = (1 - mix) * pa + mix * (RIGHT + v)
            self._dyn_add(FancyArrow(base[0], base[1], tip[0] - base[0], tip[1] - base[1],
                                     width=0.012, head_width=0.07, color="#b7950b",
                                     zorder=7, alpha=fade, length_includes_head=True))
        else:  # на правой панели: +v и −v
            self._dyn_add(FancyArrow(RIGHT[0], RIGHT[1], v[0], v[1],
                                     width=0.012, head_width=0.07, color="#b7950b",
                                     zorder=7, alpha=fade, length_includes_head=True))
            rg = np.clip((u - 0.58) / 0.18, 0, 1)
            if rg > 0:
                self._dyn_add(FancyArrow(RIGHT[0], RIGHT[1], -rg * v[0], -rg * v[1],
                                         width=0.010, head_width=0.06, color="#2e86c1",
                                         zorder=7, alpha=fade, length_includes_head=True))
            self._dyn_add(self.ax.plot(*(RIGHT + v), "o", ms=5, color="#b7950b",
                                       zorder=6, alpha=fade)[0])
            self._dyn_add(self.ax.plot(*(RIGHT - v), "o", ms=5, color="#2e86c1",
                                       zorder=6, alpha=fade)[0])

    def _draw_dots_upto(self, t, alpha=1.0):
        """Облако разностей, накопленное к моменту t."""
        pts = sample_contour(self.geom, self.n_samples)
        dots = unique_differences(pts)
        shown = 0
        acc = 0.0
        for kind, data in self.timeline:
            if kind == "pair":
                acc += self.PAIR_DUR
            elif kind == "dots":
                if t <= acc:
                    break
                take = min(self.DOT_BATCH,
                           int((t - acc) / self.DOT_BATCH_DUR * self.DOT_BATCH) + 1)
                shown = min(len(dots), shown + take)
                acc += self.DOT_BATCH_DUR
            elif kind == "fill":
                break
        if shown > 0:
            P = dots[:shown] + RIGHT
            self._dyn_add(self.ax.plot(P[:, 0], P[:, 1], "o", ms=1.6,
                                       color="#3b76c4", alpha=alpha, zorder=3)[0])

    def _draw_sweep_frame(self, pts, u):
        """Копия B скользит по контуру; позади — призраки."""
        x = u * (len(pts) - 1)
        i0 = int(np.floor(x))
        frac = x - i0
        pos = (1 - frac) * pts[i0] + frac * pts[min(i0 + 1, len(pts) - 1)]
        for g in range(0, len(pts), max(1, len(pts) // 18)):
            if g <= i0:
                self._dyn_add(self._element_patch(pts[g], alpha=0.10))
        self._dyn_add(self._element_patch(pos, alpha=0.35, lw=1.8))

    def _element_patch(self, center, alpha, lw=1.2):
        c = np.asarray(center)  # контур уже в координатах левой панели
        if self.element == "disk":
            return Circle(c, self.size, fill=False, edgecolor="#3f9d5a",
                          alpha=alpha, lw=lw, zorder=5)
        half = self.size / 2
        return Rectangle((c[0] - half, c[1] - half), self.size, self.size, fill=False,
                         edgecolor="#3f9d5a", alpha=alpha, lw=lw, zorder=5)

    # ------------------------------------------------------------ события

    def _on_preset(self, label):
        if self._guard:
            return
        name = next(k for k, v in PRESET_LABELS.items() if v == label)
        self._stop()
        self.load_preset(name)

    def _on_op(self, label):
        if self._guard:
            return
        self.op = next(o for o in OPS if OP_LABELS[o] == label)
        self._stop()
        self.recompute()

    def _on_element(self, label):
        if self._guard:
            return
        self.element = next(k for k, v in ELEMENTS.items() if v == label)
        self._stop()
        self.recompute()

    def _on_size(self, val):
        if self._guard:
            return
        self.size = float(val)
        self._stop()
        self.recompute()

    def _on_samples(self, val):
        if self._guard:
            return
        self.n_samples = int(val)
        self._build_timeline()
        if not self.playing:
            self.redraw()

    def _toggle_handles(self):
        self.show_handles = not self.show_handles
        self.redraw()

    # ------------------------------------------------------------ драг вершин

    def _connect(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, ev):
        if ev.inaxes != self.ax or ev.button != 1 or not self.show_handles:
            return
        best, bd = None, 1e9
        for ri, ring in enumerate(self.rings):
            for vi, p in enumerate(ring):
                disp = self.ax.transData.transform(p)
                d = math.hypot(ev.x - disp[0], ev.y - disp[1])
                if d < bd:
                    bd, best = d, (ri, vi)
        if best is not None and bd < 14:
            self._drag = best
            self._stop()

    def _on_motion(self, ev):
        if self._drag is None or ev.inaxes != self.ax:
            return
        ri, vi = self._drag
        data = self.ax.transData.inverted().transform((ev.x, ev.y))
        # держим вершину в пределах левой панели
        p = np.clip(np.array([data[0], data[1]]),
                    [XLIM[0] + 0.05, YLIM[0] + 0.05], [-0.05, YLIM[1] - 0.05])
        self.rings[ri][vi] = p
        self.geom = rings_to_geom(self.rings)
        self.recompute(light=True)

    def _on_release(self, ev):
        if self._drag is not None:
            self._drag = None
            self.geom = rings_to_geom(self.rings)
            self.recompute(light=False)

    # ------------------------------------------------------------ запуск

    def run(self):
        plt.show()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Transform Studio")
    ap.add_argument("--preset", default="circle", choices=PRESETS)
    ap.add_argument("--svg", default=None, help="SVG-файл для загрузки при старте")
    ap.add_argument("--selfcheck", action="store_true", help="проверить математику и выйти")
    ap.add_argument("--smoke", action="store_true", help="headless: отрендерить все режимы в PNG")
    args = ap.parse_args()

    if args.selfcheck:
        ok = selfcheck(verbose=True)
        print("SELFCHECK:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if args.smoke:
        _smoke()
        return

    app = TransformStudio(preset=args.preset, svg=args.svg)
    if not selfcheck(verbose=False):
        print("предупреждение: selfcheck не прошёл (см. --selfcheck)")
    if matplotlib.get_backend().lower() == "agg":
        print("нет графического дисплея / Tk-бэкенда: показать окно нельзя.")
        print("на Linux запусти из X/Wayland-сессии; проверь, что tkinter установлен;")
        print("либо используй --smoke / --selfcheck.")
        plt.close(app.fig)
        return
    app.run()


def _smoke():
    import itertools
    app = TransformStudio()
    for preset, op in itertools.product(PRESETS, OPS):
        app.op = op
        app.element = "disk" if (preset + op).count("s") % 2 else "square"
        app.load_preset(preset)
        app.recompute()
        app.draw_frame(app.T * 0.5)
        app.draw_frame(app.T)
        print(f"  {preset:9s} x {op:7s}: area={app.result.area:9.4f} "
              f"holes={sum(len(p.interiors) for p in iter_polygons(app.result))}")
    app.fig.savefig("/tmp/transform_studio_smoke.png", dpi=80)
    print("SMOKE: OK -> /tmp/transform_studio_smoke.png")


if __name__ == "__main__":
    main()
