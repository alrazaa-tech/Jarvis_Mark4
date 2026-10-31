"""
Iron Man faceplate for the HUD centre — a drop-in for `core.avatar.HoloAvatar`.

Same contract as the holographic head (`SPAN`, `step()`, `paint()`,
`glance()`), so `ui.py` only changes which class it builds. Everything is
QPainter paths in a unit coordinate frame — crown at y = -1, chin near
y = +1.1 — scaled to the room the HUD gives it, so like the head it needs no
OpenGL, no GPU and no new dependency.

What moves
----------
* **Eyes** are the status light: bright and steady while listening, pulsing
  while thinking, flaring with the voice while speaking, dim while asleep, off
  when muted. They take the HUD's primary colour, so the accent picker still
  retints them.
* **Mouth slit** opens with the audio level while JARVIS speaks and glows from
  inside. It never tracks the microphone.
* **Rings** around the mask carry the state accent colour and spin with the
  level — the "interface" part of the look.
"""

from __future__ import annotations

import math
import random

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen,
    QRadialGradient, QTransform,
)

# Plate colours. Fixed on purpose: red and gold *are* the look. Only the
# lights (eyes, mouth, rings, rim) follow the theme.
_RED_HI   = QColor("#e8313a")
_RED      = QColor("#a5121b")
_RED_LO   = QColor("#3d0408")
_GOLD_HI  = QColor("#ffe08a")
_GOLD     = QColor("#d1a032")
_GOLD_LO  = QColor("#5e3f0c")
_LINE_RED = QColor("#2a0205")
_LINE_GLD = QColor("#4a3108")

_TAU_OPEN = 0.03
_TAU_SHUT = 0.06
_TAU_EYE  = 0.10


def _rate(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-max(0.0, dt) / tau)


def _c(col: QColor, a: float) -> QColor:
    q = QColor(col)
    q.setAlpha(int(max(0.0, min(255.0, a))))
    return q


def _mix(a: QColor, b: QColor, f: float) -> QColor:
    f = max(0.0, min(1.0, f))
    return QColor(int(a.red() + (b.red() - a.red()) * f),
                  int(a.green() + (b.green() - a.green()) * f),
                  int(a.blue() + (b.blue() - a.blue()) * f))


def _grey(col: QColor, f: float) -> QColor:
    """Desaturate toward a dark grey by `f` — the powered-down plate."""
    g = int(col.red() * 0.30 + col.green() * 0.59 + col.blue() * 0.11)
    return _mix(col, QColor(int(g * 0.45), int(g * 0.45), int(g * 0.5)), f)


def _spline(pts: list[tuple[float, float]], closed: bool) -> QPainterPath:
    """Catmull-Rom through `pts`, as cubic Béziers."""
    n = len(pts)
    path = QPainterPath(QPointF(*pts[0]))
    segs = n if closed else n - 1
    for i in range(segs):
        p0 = pts[(i - 1) % n] if (closed or i > 0) else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (closed or i + 2 < n) else p2
        c1 = QPointF(p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = QPointF(p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        path.cubicTo(c1, c2, QPointF(*p2))
    if closed:
        path.closeSubpath()
    return path


def _mirror(half: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Right half from top-centre to bottom-centre → the whole outline."""
    return half + [(-x, y) for x, y in reversed(half[1:-1])]


def _poly(pts) -> QPainterPath:
    path = QPainterPath(QPointF(*pts[0]))
    for x, y in pts[1:]:
        path.lineTo(x, y)
    path.closeSubpath()
    return path


def _polyline(pts) -> QPainterPath:
    path = QPainterPath(QPointF(*pts[0]))
    for x, y in pts[1:]:
        path.lineTo(x, y)
    return path


# ── geometry, unit frame ────────────────────────────────────────────────────
_HELMET_HALF = [(0.0, -1.00), (0.44, -0.93), (0.73, -0.68), (0.86, -0.30),
                (0.86, 0.08), (0.78, 0.42), (0.63, 0.73), (0.40, 0.99),
                (0.0, 1.10)]

# Starts and ends at the widow's peak, so the spline leaves a sharp V there.
_PLATE_HALF = [(0.0, -0.50), (0.12, -0.61), (0.32, -0.67), (0.53, -0.58),
               (0.65, -0.36), (0.67, -0.10), (0.61, 0.20), (0.53, 0.48),
               (0.39, 0.76), (0.20, 0.94), (0.0, 1.00)]

_EYE_R = [(0.09, -0.16), (0.13, -0.235), (0.30, -0.265), (0.50, -0.275),
          (0.56, -0.215), (0.45, -0.13), (0.22, -0.105)]
_EYE_L = [(-x, y) for x, y in _EYE_R]
_EYE_C = (0.33, -0.19)

# Mouth slit: top edge from centre to the corner, which turns down.
_MOUTH_R = [(0.0, 0.615), (0.14, 0.61), (0.27, 0.60), (0.335, 0.655)]

_PLATE_LINES = [
    # cheek seams
    [(0.52, -0.10), (0.44, 0.22), (0.35, 0.60)],
    # mouth corner down to the chin
    [(0.335, 0.655), (0.25, 0.86), (0.12, 0.96)],
    # brow line
    [(0.13, -0.40), (0.30, -0.45), (0.50, -0.42)],
    # under-eye
    [(0.12, -0.06), (0.30, -0.02), (0.48, -0.06)],
]
_CENTRE_LINES = [
    [(0.0, -0.50), (0.0, -0.22)],                      # nose bridge
    [(-0.07, 0.20), (0.0, 0.26), (0.07, 0.20)],        # nose tip chevron
    [(0.0, 0.70), (0.0, 0.99)],                        # chin seam
]
_HELMET_LINES = [
    [(0.72, -0.52), (0.82, -0.20), (0.82, 0.12)],      # side seam
    [(0.72, 0.30), (0.60, 0.62), (0.44, 0.88)],        # jaw seam
    [(0.0, -1.0), (0.0, -0.50)],                       # crown ridge
    [(0.16, -0.97), (0.14, -0.62)],                    # crown ridge flank
]


def _mirrored(lines):
    return lines + [[(-x, y) for x, y in ln] for ln in lines]


class IronMask:
    """Animated Iron Man faceplate. One instance per HUD canvas."""

    # Crown (-1) to the bottom of the rings' reach; the HUD sizes us with it.
    SPAN = 2.28

    def __init__(self) -> None:
        self._helmet = _spline(_mirror(_HELMET_HALF), closed=True)
        plate_pts = _PLATE_HALF + [(-x, y) for x, y in reversed(_PLATE_HALF[:-1])]
        self._plate = _spline(plate_pts, closed=False)
        self._plate.closeSubpath()
        self._eye_r = _poly(_EYE_R)
        self._eye_l = _poly(_EYE_L)
        self._eyes = QPainterPath(self._eye_r)
        self._eyes.addPath(self._eye_l)
        # Recess around each eye — the same slit, grown about its centre.
        self._sockets = QPainterPath()
        for sx in (1.0, -1.0):
            t = QTransform()
            t.translate(sx * _EYE_C[0], _EYE_C[1])
            t.scale(1.28, 1.75)
            t.translate(-sx * _EYE_C[0], -_EYE_C[1])
            self._sockets.addPath(t.map(self._eye_r if sx > 0 else self._eye_l))

        self._plate_lines = QPainterPath()
        for ln in _mirrored(_PLATE_LINES) + _CENTRE_LINES:
            self._plate_lines.addPath(_polyline(ln))
        self._helmet_lines = QPainterPath()
        for ln in _mirrored(_HELMET_LINES):
            self._helmet_lines.addPath(_polyline(ln))

        self._t = 0.0
        self._mouth = 0.0
        self._eye = 0.0           # 0..1 eye brightness, smoothed
        self._level = 0.0         # smoothed audio level, for the rings
        self._spin = 0.0          # integrated ring rotation
        self._scan = -1.2
        self._boot = 0.0          # 0..1 power-up on first frames
        self._flicker = 0.0
        self._flicker_at = 4.0
        self._speaking = False
        self._muted = False
        self._state = ""
        self._glance = None       # (dx, dy, until_t)
        self._gx = 0.0
        self._gy = 0.0

    # ── animation ───────────────────────────────────────────────────────────

    def step(self, dt: float, amp: float, speaking: bool = False,
             muted: bool = False, state: str = "",
             v_open: float | None = None, v_wide: float = 0.0,
             v_level: float | None = None,
             v_seq: list | None = None, v_hop: float = 0.02) -> None:
        dt = max(0.0, min(0.1, dt))
        self._t += dt
        self._speaking = speaking
        self._muted = muted
        self._state = state or ""
        amp = max(0.0, min(1.0, float(amp or 0.0)))
        self._boot = min(1.0, self._boot + dt / 1.4)

        # Mouth: only ever JARVIS's own voice.
        if speaking and not muted:
            lvl = v_level if v_level is not None else amp
            opn = v_open if v_open is not None else lvl
            tgt = max(0.0, min(1.0, 0.25 * lvl + 1.1 * opn))
        else:
            tgt = 0.0
        tau = _TAU_OPEN if tgt > self._mouth else _TAU_SHUT
        self._mouth += (tgt - self._mouth) * _rate(dt, tau)

        # Eyes: the status light.
        st = self._state
        if muted:
            e = 0.08
        elif speaking:
            e = 0.82 + 0.18 * amp
        elif st in ("THINKING", "PROCESSING"):
            e = 0.55 + 0.35 * (0.5 + 0.5 * math.sin(self._t * 6.0))
        elif st == "SLEEPING":
            e = 0.18 + 0.07 * math.sin(self._t * 1.3)
        elif st == "LISTENING":
            e = 0.80 + 0.20 * amp
        else:
            e = 0.55 + 0.08 * math.sin(self._t * 2.0)
        self._eye += (e - self._eye) * _rate(dt, _TAU_EYE)

        # The odd power flicker, so it reads as a machine rather than a sticker.
        if self._t >= self._flicker_at:
            self._flicker = 1.0
            self._flicker_at = self._t + random.uniform(5.0, 11.0)
        self._flicker = max(0.0, self._flicker - dt * 9.0)

        self._level += (amp - self._level) * _rate(dt, 0.08)
        self._spin += dt * (0.25 + (1.6 if speaking else 0.6) * self._level)
        self._scan += dt * 0.55
        if self._scan > 1.35:
            self._scan = -1.2

        gx = gy = 0.0
        if self._glance is not None:
            dx, dy, until = self._glance
            if self._t < until:
                gx, gy = dx, dy
            else:
                self._glance = None
        self._gx += (gx - self._gx) * _rate(dt, 0.25)
        self._gy += (gy - self._gy) * _rate(dt, 0.25)

    def glance(self, dx: float, dy: float, hold: float = 1.1) -> None:
        """Turn toward a point for `hold` seconds; dx, dy in -1..1."""
        self._glance = (max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)),
                        self._t + max(0.1, hold))

    # ── drawing ─────────────────────────────────────────────────────────────

    def paint(self, p: QPainter, cx: float, cy: float, r: float,
              primary: QColor, accent: QColor, bg: QColor) -> None:
        if r <= 4:
            return
        p.save()
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            mid_y = cy + (self.SPAN / 2.0 - 1.0) * r
            self._paint_rings(p, cx, mid_y, r, primary, accent)

            yaw = 0.035 * math.sin(self._t * 0.45) + 0.06 * self._gx
            bob = 0.012 * math.sin(self._t * 0.9) + 0.03 * self._gy
            t = QTransform()
            t.translate(cx + yaw * r * 0.6, cy + 0.04 * r + bob * r)
            t.scale(r * 0.94 * math.cos(yaw), r * 0.94)
            p.setTransform(t, True)
            self._paint_mask(p, primary, accent, bg)
        finally:
            p.restore()

    def _paint_rings(self, p, cx, cy, r, primary, accent):
        R = r * (self.SPAN / 2.0) * 1.02
        boot = self._boot
        a_pri = 70 * boot
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Faint full circle and tick marks.
        p.setPen(QPen(_c(primary, a_pri * 0.6), max(1.0, r * 0.006)))
        p.drawEllipse(QPointF(cx, cy), R, R)
        p.setPen(QPen(_c(primary, a_pri), max(1.0, r * 0.008)))
        n = 72
        for i in range(n):
            ang = (i / n) * math.tau + self._spin * 0.15
            l0 = R * (0.955 if i % 6 else 0.93)
            ca, sa = math.cos(ang), math.sin(ang)
            p.drawLine(QPointF(cx + ca * l0, cy + sa * l0),
                       QPointF(cx + ca * R * 0.985, cy + sa * R * 0.985))

        # Spinning accent arcs — state colour, thicken with the voice.
        lw = max(1.5, r * (0.014 + 0.02 * self._level))
        for k, (rr, span, speed, alpha) in enumerate((
                (1.07, 70, 1.0, 190), (1.07, 40, 1.0, 120),
                (0.90, 110, -0.7, 110), (1.14, 25, 1.6, 150))):
            pen = QPen(_c(accent, alpha * boot), lw if k != 2 else lw * 0.6)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            rad = R * rr
            rect = QRectF(cx - rad, cy - rad, rad * 2, rad * 2)
            start = (self._spin * 57.3 * speed + k * 137.0) % 360.0
            p.drawArc(rect, int(start * 16), int(span * 16))
            p.drawArc(rect, int((start + 180.0) * 16), int(span * 16))

    def _paint_mask(self, p, primary, accent, bg):
        off = 1.0 if self._muted else 0.0
        flick = 1.0 - 0.55 * self._flicker
        eye = max(0.0, min(1.0, self._eye * flick * self._boot))

        red_hi, red, red_lo = (_grey(c, off) for c in (_RED_HI, _RED, _RED_LO))
        g_hi, g, g_lo = (_grey(c, off) for c in (_GOLD_HI, _GOLD, _GOLD_LO))
        eye_col = _mix(primary, QColor(255, 255, 255), 0.55)
        if self._muted:
            eye_col = QColor("#ff3355")

        # Back glow, so the mask sits in light rather than on a void.
        halo = QRadialGradient(QPointF(0, 0.0), 1.45)
        halo.setColorAt(0.0, _c(primary, 55 * eye))
        halo.setColorAt(0.6, _c(primary, 18 * eye))
        halo.setColorAt(1.0, _c(primary, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(QPointF(0, 0.05), 1.45, 1.45)

        # ── helmet shell ────────────────────────────────────────────────────
        shell = QLinearGradient(-0.9, 0, 0.9, 0)
        shell.setColorAt(0.0, red_lo)
        shell.setColorAt(0.28, red)
        shell.setColorAt(0.42, red_hi)
        shell.setColorAt(0.70, red)
        shell.setColorAt(1.0, red_lo)
        p.setBrush(QBrush(shell))
        p.setPen(QPen(_LINE_RED, 0.02))
        p.drawPath(self._helmet)

        shine = QRadialGradient(QPointF(-0.30, -0.75), 0.75)
        shine.setColorAt(0.0, _c(QColor(255, 210, 210), 110 * (1 - off)))
        shine.setColorAt(1.0, _c(QColor(255, 210, 210), 0))
        p.setBrush(QBrush(shine))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(self._helmet)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(_LINE_RED, 220), 0.012))
        p.drawPath(self._helmet_lines)

        # ── gold faceplate ──────────────────────────────────────────────────
        plate = QRadialGradient(QPointF(-0.12, -0.28), 1.25)
        plate.setColorAt(0.0, g_hi)
        plate.setColorAt(0.35, g)
        plate.setColorAt(0.85, g_lo)
        plate.setColorAt(1.0, _mix(g_lo, QColor(0, 0, 0), 0.4))
        p.setBrush(QBrush(plate))
        p.setPen(QPen(_mix(red_lo, QColor(0, 0, 0), 0.3), 0.022))
        p.drawPath(self._plate)

        # Edge shading down the plate's sides — it reads as curved metal.
        side = QLinearGradient(-0.7, 0, 0.7, 0)
        side.setColorAt(0.0, _c(QColor(0, 0, 0), 120))
        side.setColorAt(0.25, _c(QColor(0, 0, 0), 0))
        side.setColorAt(0.75, _c(QColor(0, 0, 0), 0))
        side.setColorAt(1.0, _c(QColor(0, 0, 0), 140))
        p.setBrush(QBrush(side))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(self._plate)

        p.setBrush(QBrush(_c(_mix(g_lo, QColor(0, 0, 0), 0.35), 230)))
        p.drawPath(self._sockets)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(_LINE_GLD, 210), 0.011))
        p.drawPath(self._plate_lines)

        # ── eyes ────────────────────────────────────────────────────────────
        for (ex, ey) in ((_EYE_C[0], _EYE_C[1]), (-_EYE_C[0], _EYE_C[1])):
            flare = QRadialGradient(QPointF(ex, ey), 0.55)
            flare.setColorAt(0.0, _c(eye_col, 120 * eye))
            flare.setColorAt(1.0, _c(eye_col, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(flare))
            p.drawEllipse(QPointF(ex, ey), 0.55, 0.55)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for w, a in ((0.09, 40), (0.055, 70), (0.03, 120)):
            pen = QPen(_c(eye_col, a * eye), w)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPath(self._eyes)
        core = QLinearGradient(0, -0.28, 0, -0.10)
        core.setColorAt(0.0, _c(QColor(255, 255, 255), 255 * eye))
        core.setColorAt(1.0, _c(eye_col, 255 * max(0.25, eye)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(core) if eye > 0.02 else QBrush(QColor(10, 6, 6)))
        p.drawPath(self._eyes)

        # ── mouth ───────────────────────────────────────────────────────────
        gap = 0.012 + 0.075 * self._mouth
        top = [(-x, y) for x, y in reversed(_MOUTH_R)] + _MOUTH_R[1:]
        bot = [(x, y + gap * (1.0 - 0.55 * (abs(x) / 0.335) ** 2))
               for x, y in reversed(top)]
        mouth = _poly(top + bot)
        p.setBrush(QBrush(QColor(8, 3, 3)))
        p.setPen(QPen(_c(_LINE_GLD, 230), 0.012))
        p.drawPath(mouth)
        m_glow = (0.15 + 0.85 * self._mouth) * eye if self._speaking else 0.1 * eye
        if m_glow > 0.02:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_c(eye_col, 70 * m_glow), 0.05))
            p.drawPath(mouth)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_c(eye_col, 210 * m_glow)))
            p.drawPath(mouth)

        # ── holographic finish: rim light and a scan sweep ──────────────────
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(primary, 90 * self._boot), 0.012))
        p.drawPath(self._helmet)

        if not self._muted:
            p.save()
            p.setClipPath(self._helmet)
            band = QLinearGradient(0, self._scan - 0.12, 0, self._scan + 0.12)
            band.setColorAt(0.0, _c(primary, 0))
            band.setColorAt(0.5, _c(primary, 55))
            band.setColorAt(1.0, _c(primary, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(band))
            p.drawRect(QRectF(-1.0, self._scan - 0.12, 2.0, 0.24))
            p.restore()
