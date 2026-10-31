"""
JARVIS particle orb for the HUD centre — a drop-in for `core.avatar.HoloAvatar`.

Same contract as the holographic head and the old faceplate (`SPAN`, `step()`,
`paint()`, `glance()`), so `ui.py` only changes which class it builds. Pure
QPainter + numpy, no OpenGL and no new dependency.

What moves
----------
* **Shell** — a few hundred "atoms" on a sphere, linked by a faint network,
  turning slowly in 3D. While JARVIS speaks (or the mic hears you) every atom
  ripples radially with the audio level, so the surface moves with the voice.
* **Orbit rings** — three tilted swirls of particles circling the shell; they
  speed up and flutter with the level.
* **Core** — a dense, white-hot cluster that flares with the voice.
* **Size** — the whole orb swells while speaking and breathes when idle.
* **Sparks** — loud syllables throw particles off the surface.
* **HUD frame** — rotating arcs and a tick ring whose ticks grow like a
  spectrum with the level.

Colours are fixed amber/gold (that *is* the look); muted switches to the
HUD's muted colour.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF, QRadialGradient

# Hot centre → deep ember. Index 0 is the brightest.
_AMBER_PALETTE = (
    (255, 244, 214),
    (255, 198, 96),
    (255, 146, 40),
    (214, 88, 16),
)

_PERSPECTIVE = 3.2


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class JarvisOrb:
    """Rotating, voice-reactive particle orb. One instance per HUD canvas."""

    # Unit frame spans -1..+1 vertically; the HUD centres us in its band.
    SPAN = 2.0

    def __init__(self) -> None:
        rng = np.random.default_rng(7)

        # ── shell: jittered Fibonacci sphere ─────────────────────────────────
        n = 520
        i = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n)
        th = math.pi * (1.0 + 5 ** 0.5) * i
        sh = np.stack([np.cos(th) * np.sin(phi), np.cos(phi),
                       np.sin(th) * np.sin(phi)], axis=1)
        sh += rng.normal(0.0, 0.035, sh.shape)
        sh /= np.linalg.norm(sh, axis=1, keepdims=True)
        self._sh = sh
        self._sh_r = rng.uniform(0.88, 1.04, n)
        self._sh_ph = rng.uniform(0.0, 2 * math.pi, n)
        self._sh_w = rng.uniform(3.0, 9.0, n)
        self._sh_lat = np.arcsin(np.clip(sh[:, 1], -1, 1))
        self._sh_lon = np.arctan2(sh[:, 2], sh[:, 0])
        self._sh_b = rng.uniform(0.45, 1.0, n)

        # Network links between near neighbours, thinned so it reads as a web
        # rather than a mesh.
        cos_sim = sh @ sh.T
        np.fill_diagonal(cos_sim, -2.0)
        nb = np.argsort(-cos_sim, axis=1)[:, :3]
        pairs = set()
        lim = math.cos(0.26)
        for a in range(n):
            for b in nb[a]:
                if cos_sim[a, b] > lim and rng.random() < 0.55:
                    pairs.add((min(a, int(b)), max(a, int(b))))
        self._pairs = np.array(sorted(pairs), dtype=int).reshape(-1, 2)

        # ── orbit rings: (radius, tilt_x, tilt_z, count, speed) ──────────────
        self._rings = []
        for rad, tx, tz, cnt, spd in ((1.24, 1.15, 0.25, 150, 0.85),
                                      (1.38, -0.55, 0.75, 120, -0.55),
                                      (1.12, 0.30, -1.05, 100, 1.25)):
            # Particles bunch into a few long arcs, like streaks of a swirl.
            centres = rng.uniform(0, 2 * math.pi, 3)
            ang = (centres[rng.integers(0, 3, cnt)]
                   + rng.normal(0.0, 0.9, cnt))
            self._rings.append(dict(
                rad=rad, tilt=_rot_x(tx) @ _rot_z(tz), spd=spd, ang=ang,
                dr=rng.normal(0.0, 0.03, cnt), dy=rng.normal(0.0, 0.022, cnt),
                ph=rng.uniform(0, 2 * math.pi, cnt),
                b=rng.uniform(0.4, 1.0, cnt)))

        # ── core: dense cluster in the middle ────────────────────────────────
        m = 150
        d = rng.normal(0.0, 1.0, (m, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self._core = d * (rng.uniform(0.0, 1.0, m) ** 1.6 * 0.42)[:, None]
        self._core_ph = rng.uniform(0, 2 * math.pi, m)
        self._core_w = rng.uniform(4.0, 11.0, m)

        # ── HUD frame tick ring ──────────────────────────────────────────────
        self._tick_n = 120
        self._tick_ph = rng.uniform(0, 2 * math.pi, self._tick_n)

        # ── animation state ─────────────────────────────────────────────────
        self._t = 0.0
        self._boot = 0.0
        self._amp = 0.0
        self._yaw = 0.0
        self._hud_rot = 0.0
        self._size = 1.0
        self._speaking = False
        self._muted = False
        self._state = ""
        self._sparks: list[list[float]] = []   # [x, y, vx, vy, life, max_life]
        self._spark_acc = 0.0
        self._gx = self._gy = 0.0
        self._glance = None

    # ── contract ────────────────────────────────────────────────────────────
    def step(self, dt: float, amp: float, speaking: bool = False,
             muted: bool = False, state: str = "", **_unused) -> None:
        dt = min(0.1, max(0.0, float(dt)))
        t = self._t = self._t + dt
        self._boot = min(1.0, self._boot + dt / 1.4)
        self._speaking, self._muted, self._state = speaking, muted, state

        # Fast attack, slower release so syllables punch but don't flicker.
        k = 22.0 if amp > self._amp else 9.0
        self._amp += (amp - self._amp) * min(1.0, dt * k)
        a = 0.0 if muted else self._amp

        thinking = state in ("THINKING", "PROCESSING")
        if muted:
            spin = 0.07
        else:
            spin = (0.22 + (0.60 if thinking else 0.0)
                    + (0.25 if speaking else 0.0) + a * 1.6)
        self._yaw += spin * dt
        self._hud_rot += (0.18 + spin * 0.4) * dt

        if muted:
            tgt = 0.88
        else:
            tgt = (1.0 + 0.025 * math.sin(t * 1.3)
                   + (0.10 if speaking else 0.0)
                   + (0.03 if state == "LISTENING" else 0.0)
                   + a * 0.26)
        rate = 10.0 if tgt > self._size else 3.5
        self._size += (tgt - self._size) * min(1.0, dt * rate)

        # Sparks thrown off the surface on loud moments.
        if a > 0.08 and not muted:
            self._spark_acc += a * 110.0 * dt
            while self._spark_acc >= 1.0 and len(self._sparks) < 220:
                self._spark_acc -= 1.0
                ang = random.uniform(0, 2 * math.pi)
                ca, sa = math.cos(ang), math.sin(ang)
                r0 = self._size * random.uniform(0.9, 1.05)
                v = random.uniform(0.5, 1.5) * (1.0 + a)
                life = random.uniform(0.5, 1.1)
                self._sparks.append([ca * r0, sa * r0, ca * v, sa * v, life, life])
        else:
            self._spark_acc = 0.0
        alive = []
        drag = 0.985 ** (dt * 60.0)
        for s in self._sparks:
            s[4] -= dt
            if s[4] > 0:
                s[0] += s[2] * dt
                s[1] += s[3] * dt
                s[2] *= drag
                s[3] *= drag
                alive.append(s)
        self._sparks = alive

        # Glance: lean the orb toward a point for a moment.
        gx = gy = 0.0
        if self._glance is not None:
            if t < self._glance[2]:
                gx, gy = self._glance[0], self._glance[1]
            else:
                self._glance = None
        self._gx += (gx - self._gx) * min(1.0, dt * 4.0)
        self._gy += (gy - self._gy) * min(1.0, dt * 4.0)

    def glance(self, dx: float, dy: float, hold: float = 1.1) -> None:
        """Lean toward a point for `hold` seconds; dx, dy in -1..1."""
        self._glance = (max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)),
                        self._t + max(0.1, hold))

    def paint(self, p: QPainter, cx: float, cy: float, r: float,
              primary: QColor, accent: QColor, bg: QColor) -> None:
        if r <= 4:
            return
        p.save()
        try:
            self._paint(p, cx, cy, r * 0.52, primary)
        finally:
            p.restore()

    # ── drawing ─────────────────────────────────────────────────────────────
    def _palette(self, primary: QColor):
        if self._muted:
            pr, pg, pb = primary.red(), primary.green(), primary.blue()
            return tuple((int(pr * f), int(pg * f), int(pb * f))
                         for f in (1.0, 0.8, 0.6, 0.45))
        return _AMBER_PALETTE

    def _paint(self, p: QPainter, cx: float, cy: float, R: float,
               primary: QColor) -> None:
        t = self._t
        a = 0.0 if self._muted else self._amp
        size = self._size
        boot = self._boot
        pal = self._palette(primary)
        px = max(0.6, R / 140.0)          # pixel scale for pen widths

        def col(i: int, alpha: float) -> QColor:
            c = pal[i]
            return QColor(c[0], c[1], c[2], int(max(0, min(255, alpha))))

        # 1. Atmosphere — a soft body of light behind everything.
        glow = boot * (0.55 + 0.9 * a + (0.15 if self._speaking else 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        gr = R * 1.9 * size
        g = QRadialGradient(cx, cy, gr)
        g.setColorAt(0.00, col(1, 95 * glow))
        g.setColorAt(0.35, col(2, 45 * glow))
        g.setColorAt(0.70, col(3, 14 * glow))
        g.setColorAt(1.00, col(3, 0))
        p.setBrush(QBrush(g))
        p.drawEllipse(QRectF(cx - gr, cy - gr, gr * 2, gr * 2))
        p.setBrush(Qt.BrushStyle.NoBrush)

        # 2. HUD frame — grows a little with the orb, much less than it.
        hs = 1.0 + (size - 1.0) * 0.3
        self._paint_frame(p, cx, cy, R * hs, a, boot, col, px)

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        # 3. 3D particles.
        pitch = 0.40 + 0.07 * math.sin(t * 0.37) + 0.25 * self._gy
        yaw = self._yaw + 0.45 * self._gx
        M = _rot_x(pitch) @ _rot_y(yaw)

        # Shell, rippling with the voice.
        ripple = (0.20 * np.sin(self._sh_lat * 6.0 + t * 8.0 + self._sh_ph * 0.3)
                  + 0.16 * np.sin(self._sh_lon * 5.0 - t * 6.5)
                  + 0.24 * np.sin(t * self._sh_w + self._sh_ph))
        idle = 0.018 * np.sin(t * self._sh_w * 0.25 + self._sh_ph)
        rad = self._sh_r * size * (1.0 + 0.6 * a * ripple + idle)
        shell = (self._sh * rad[:, None]) @ M.T

        parts = [shell]
        bright = [self._sh_b]
        tone = [np.full(len(shell), 1)]

        # Rings.
        for ring in self._rings:
            ang = ring["ang"] + t * ring["spd"] * (1.0 + a * 1.6)
            rr = ring["rad"] * size * (1.0 + ring["dr"]
                                       + a * 0.10 * np.sin(ang * 7.0 + t * 11.0 + ring["ph"]))
            local = np.stack([np.cos(ang) * rr, ring["dy"] * size,
                              np.sin(ang) * rr], axis=1)
            parts.append(local @ (M @ ring["tilt"]).T)
            bright.append(ring["b"])
            tone.append(np.full(len(ang), 2))

        # Core — turns the other way, faster.
        Mc = _rot_x(pitch * 0.6) @ _rot_y(-self._yaw * 1.8)
        pulse = 1.0 + a * 0.7 * np.sin(t * self._core_w + self._core_ph)
        core = (self._core * (math.sqrt(size) * pulse)[:, None]) @ Mc.T
        parts.append(core)
        bright.append(np.full(len(core), 1.0))
        tone.append(np.full(len(core), 0))

        X = np.concatenate(parts)
        B = np.concatenate(bright)
        T = np.concatenate(tone)

        k = _PERSPECTIVE / (_PERSPECTIVE - X[:, 2])
        sx = cx + X[:, 0] * R * k
        sy = cy + X[:, 1] * R * k
        depth = np.clip((X[:, 2] + 1.5) / 3.0, 0.0, 1.0)
        lum = B * (0.22 + 0.78 * depth) * boot * (0.8 + 0.6 * a)

        # 3a. Network links on the shell, faded with depth.
        if len(self._pairs):
            pa, pb = self._pairs[:, 0], self._pairs[:, 1]
            dl = (depth[pa] + depth[pb]) * 0.5
            front = dl > 0.5
            x1, y1, x2, y2 = (sx[pa].tolist(), sy[pa].tolist(),
                              sx[pb].tolist(), sy[pb].tolist())
            for mask, alpha in ((~front, 22), (front, 55)):
                idx = np.nonzero(mask)[0].tolist()
                if not idx:
                    continue
                p.setPen(QPen(col(2, alpha * boot * (1.0 + 1.2 * a)), 0.8 * px))
                p.drawLines([QLineF(x1[j], y1[j], x2[j], y2[j]) for j in idx])

        # 3b. Atoms, bucketed by brightness so each bucket is one draw call.
        lvl = np.clip((lum * 4.0).astype(int), 0, 3)
        sxl, syl = sx.tolist(), sy.tolist()
        for b in range(4):
            for tn in (0, 1, 2):
                idx = np.nonzero((lvl == b) & (T == tn))[0].tolist()
                if not idx:
                    continue
                poly = QPolygonF([QPointF(sxl[j], syl[j]) for j in idx])
                ci = 0 if tn == 0 else (1 if b >= 2 else 2)
                if b >= 2:            # soft halo on the bright ones
                    pen = QPen(col(ci + 1 if ci < 3 else 3, 26 + 14 * a),
                               (4.2 + b * 1.2) * px)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    p.setPen(pen)
                    p.drawPoints(poly)
                pen = QPen(col(ci, 70 + b * 50), (1.1 + b * 0.45) * px)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawPoints(poly)

        # 4. Sparks — short streaks flying outward.
        if self._sparks:
            streaks = []
            for x, y, vx, vy, life, ml in self._sparks:
                f = life / ml
                x0, y0 = cx + x * R, cy + y * R
                streaks.append((f, QLineF(x0, y0, x0 - vx * R * 0.06,
                                          y0 - vy * R * 0.06)))
            for lo, hi, alpha in ((0.0, 0.5, 70), (0.5, 1.01, 170)):
                lines = [ln for f, ln in streaks if lo <= f < hi]
                if lines:
                    pen = QPen(col(1, alpha), 1.4 * px)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    p.setPen(pen)
                    p.drawLines(lines)

        # 5. White-hot centre.
        p.setPen(Qt.PenStyle.NoPen)
        cr = R * 0.45 * (1.0 + a * 0.8) * math.sqrt(size)
        g = QRadialGradient(cx, cy, cr)
        heat = boot * (0.7 + 0.6 * a)
        g.setColorAt(0.00, col(0, 230 * heat))
        g.setColorAt(0.18, col(1, 150 * heat))
        g.setColorAt(0.50, col(2, 45 * heat))
        g.setColorAt(1.00, col(3, 0))
        p.setBrush(QBrush(g))
        p.drawEllipse(QRectF(cx - cr, cy - cr, cr * 2, cr * 2))

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    def _paint_frame(self, p: QPainter, cx: float, cy: float, R: float,
                     a: float, boot: float, col, px: float) -> None:
        t = self._t
        rot = math.degrees(self._hud_rot)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Faint guide circles.
        for rr, alpha in ((1.50, 40), (1.66, 26)):
            rad = R * rr
            p.setPen(QPen(col(2, alpha * boot), 1.0 * px))
            p.drawEllipse(QRectF(cx - rad, cy - rad, rad * 2, rad * 2))

        # Rotating arcs, in opposite directions.
        for rr, span, count, dirn, alpha, wid in ((1.56, 70, 3, +1, 150, 2.0),
                                                  (1.62, 28, 5, -1, 90, 1.2),
                                                  (1.44, 120, 2, -1, 60, 1.0)):
            rad = R * rr
            pen = QPen(col(1, alpha * boot * (0.8 + 0.5 * a)), wid * px)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            box = QRectF(cx - rad, cy - rad, rad * 2, rad * 2)
            base = (rot * dirn * (1.0 + rr * 0.3)) % 360.0
            for s in range(count):
                p.drawArc(box, int((base + s * 360.0 / count) * 16), int(span * 16))

        # Tick ring — doubles as a spectrum: ticks reach out with the voice.
        n = self._tick_n
        r0 = R * 1.72
        ticks_lo, ticks_hi = [], []
        for i in range(n):
            ang = 2 * math.pi * i / n + self._hud_rot * 0.25
            ca, sa = math.cos(ang), math.sin(ang)
            env = abs(math.sin(i * 0.37 + t * 5.0 + self._tick_ph[i] * 0.5))
            ln = R * (0.025 + (0.06 if i % 10 == 0 else 0.0) + a * 0.14 * env)
            line = QLineF(cx + ca * r0, cy + sa * r0,
                          cx + ca * (r0 + ln), cy + sa * (r0 + ln))
            (ticks_hi if ln > R * 0.07 else ticks_lo).append(line)
        p.setPen(QPen(col(2, 70 * boot), 1.0 * px))
        p.drawLines(ticks_lo)
        if ticks_hi:
            p.setPen(QPen(col(1, 170 * boot), 1.4 * px))
            p.drawLines(ticks_hi)

        # Corner brackets framing the orb.
        e, arm = R * 1.95, R * 0.28
        p.setPen(QPen(col(2, 90 * boot), 1.3 * px))
        for sx_, sy_ in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x, y = cx + sx_ * e, cy + sy_ * e
            p.drawLine(QLineF(x, y, x - sx_ * arm, y))
            p.drawLine(QLineF(x, y, x, y - sy_ * arm))
