import argparse, math, random
from collections import deque
import cv2
import numpy as np
import mediapipe as mp

# ------------------------------------------------------------------ constants
RED, BLUE, MAGENTA = (40, 40, 235), (235, 60, 20), (220, 0, 220)   # BGR colours
NAVY = (110, 20, 40)                                                # scribble colour (BGR)
TWOS = 3            # redraw the wobble every 3rd frame at 30fps = 10 "drawings"/sec (animation on 3s)
N_PTS = 30          # points on the blob outline

# --- flick / boom settings (tune these if the boom fires too early or never)
DOT_TIP = 12        # landmark that carries the small dot: 12 = middle fingertip (use 8 for index)
FLICK_CLOSED = 0.50 # thumb<->finger distance (x palm size) below this = "pinched / cocked"
FLICK_OPEN = 0.80   # ...and above this = "released"  -> release counts as a flick
FLICK_FRAMES = 3    # flick must hold this many frames in a row (ignores tracker glitches)
FLICK_SPEED = 0.55  # backup trigger: fingertip speed relative to the wrist (x palm size per frame)


# ------------------------------------------------------------------ helpers
def ease_out(t):                       # fast start, slow end (0..1 -> 0..1)
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


class Spring:
    """Follows a target with overshoot -> gives the 'floaty / jelly' lag of the video."""
    def __init__(self, p, k=0.18, damp=0.72):
        self.p = np.array(p, float); self.v = np.zeros(2); self.k, self.d = k, damp

    def step(self, target):
        self.v = (self.v + (np.array(target, float) - self.p) * self.k) * self.d
        self.p = self.p + self.v
        return self.p


# ------------------------------------------------------------------ blob
class Blob:
    def __init__(self, colour, dot=False):
        self.colour, self.dot = colour, dot
        self.pos = Spring((0, 0))
        self.r = 0.0
        self.noise = np.random.uniform(-1, 1, N_PTS)   # current "drawing" of the wobble
        self.frame = 0
        self.alpha = 1.0
        self.blur = 0

    def update(self, target, radius):
        self.frame += 1
        if self.frame % TWOS == 0:                      # new wobble only every TWOS frames
            self.noise = np.random.uniform(-1, 1, N_PTS)
        self.pos.step(target)
        self.r = lerp(self.r, radius, 0.25)             # smooth grow / shrink

    def draw(self, img):
        if self.r < 1.5 or self.alpha <= 0.01:
            return
        v = self.pos.v
        speed = float(np.linalg.norm(v))
        ang = math.atan2(v[1], v[0]) if speed > 0.5 else 0.0
        stretch = 1 + min(speed / 70.0, 0.45)            # squash & stretch along velocity
        pts = []
        for i in range(N_PTS):
            a = 2 * math.pi * i / N_PTS
            rad = self.r * (1 + 0.11 * self.noise[i])   # wobbly radius = hand-drawn outline
            x, y = math.cos(a) * rad, math.sin(a) * rad
            # rotate into velocity frame, stretch x, squash y, rotate back
            xr = x * math.cos(-ang) - y * math.sin(-ang)
            yr = x * math.sin(-ang) + y * math.cos(-ang)
            xr *= stretch; yr /= math.sqrt(stretch)
            # teardrop tail: points pointing backwards get pulled out when moving fast
            if speed > 22 and xr < 0:
                xr *= 1 + min((speed - 22) / 40.0, 0.9) * (abs(yr) < rad * 0.35)
            xw = xr * math.cos(ang) - yr * math.sin(ang)
            yw = xr * math.sin(ang) + yr * math.cos(ang)
            pts.append((self.pos.p[0] + xw, self.pos.p[1] + yw))
        layer = img.copy()
        cv2.fillPoly(layer, [np.int32(pts)], self.colour, cv2.LINE_AA)
        if self.dot:                                    # white pupil of the blue blob
            cv2.circle(layer, tuple(np.int32(self.pos.p)), max(2, int(self.r * 0.12)), (255, 255, 255), -1, cv2.LINE_AA)
        if self.blur:
            layer = cv2.GaussianBlur(layer, (0, 0), self.blur)
        cv2.addWeighted(layer, self.alpha, img, 1 - self.alpha, 0, img)


# ------------------------------------------------------------------ scribbles / speed lines
class Stroke:
    """A thin curved navy line that lives ~0.25s (the 'smear' arcs around the blob)."""
    def __init__(self, centre, colour=NAVY, life=8, burst=False):
        self.life = self.max = life; self.colour = colour
        a0 = random.uniform(0, 2 * math.pi)
        r0 = random.uniform(30, 90) if not burst else random.uniform(15, 40)
        r1 = r0 + (random.uniform(60, 160) if not burst else random.uniform(80, 220))
        p0 = (centre[0] + math.cos(a0) * r0, centre[1] + math.sin(a0) * r0)
        p2 = (centre[0] + math.cos(a0 + random.uniform(-.7, .7)) * r1,
              centre[1] + math.sin(a0 + random.uniform(-.7, .7)) * r1)
        pm = ((p0[0] + p2[0]) / 2 + random.uniform(-40, 40), (p0[1] + p2[1]) / 2 + random.uniform(-40, 40))
        t = np.linspace(0, 1, 20)[:, None]                       # quadratic bezier
        self.pts = np.int32((1 - t) ** 2 * p0 + 2 * (1 - t) * t * pm + t ** 2 * p2)

    def draw(self, img):
        grow = 1 - self.life / self.max                          # line "draws on" over its life
        n = max(2, int(len(self.pts) * min(1, grow * 2 + 0.2)))
        cv2.polylines(img, [self.pts[:n]], False, self.colour, 2, cv2.LINE_AA)
        self.life -= 1
        return self.life > 0


# ------------------------------------------------------------------ BOOM (dot blows up)
class Boom:
    """Dot inflates fast -> POPS -> white flash + shockwave ring + radial lines + confetti + screen shake."""
    POP, LIFE = 7, 34            # frame of the pop, total frames of the effect

    def __init__(self, c, R):
        self.c = np.array(c, float); self.R = max(float(R), 60.0); self.t = 0; self.done = False
        R = self.R
        self.blob = Blob(MAGENTA); self.blob.pos = Spring(self.c)          # the inflating dot
        self.rays = []                                                     # radial burst lines
        for i in range(22):
            self.rays.append(dict(a=2 * math.pi * i / 22 + random.uniform(-.12, .12),
                                  L=R * random.uniform(1.0, 2.2), curve=random.uniform(-.25, .25),
                                  col=random.choice([NAVY, NAVY, (0, 0, 0), (150, 20, 110)]),
                                  w=random.choice([2, 3, 3])))
        self.parts = []                                                    # confetti dots flying out
        for _ in range(45):
            a = random.uniform(0, 2 * math.pi); sp = random.uniform(0.08, 0.30) * R
            self.parts.append(dict(p=self.c.copy(), v=np.array([math.cos(a), math.sin(a)]) * sp,
                                   r=random.uniform(0.05, 0.14) * R, col=random.choice([RED, BLUE, MAGENTA, MAGENTA])))

    def update(self):
        self.t += 1
        if self.t >= self.POP:                                             # confetti physics: drag + shrink
            for q in self.parts:
                q["p"] = q["p"] + q["v"]; q["v"] = q["v"] * 0.85; q["r"] *= 0.93
        if self.t > self.LIFE:
            self.done = True

    def shake(self):                                                       # camera shake right after the pop
        q = self.t - self.POP
        if q < 0 or q > 8:
            return 0, 0
        amp = 16 * (1 - q / 9)
        return int(random.uniform(-amp, amp)), int(random.uniform(-amp, amp))

    def draw(self, img):
        t, R = self.t, self.R
        if t < self.POP:                                                   # phase 1: dot swells like a balloon
            self.blob.frame += 1
            if self.blob.frame % TWOS == 0:
                self.blob.noise = np.random.uniform(-1, 1, N_PTS)
            self.blob.r = lerp(6, R * 0.95, ease_out((t + 1) / self.POP))
            self.blob.draw(img)
            return
        q = t - self.POP                                                   # frames since the pop
        a = max(0.0, 0.55 - 0.11 * q)                                      # white flash, fades in ~5 frames
        if a > 0:
            cv2.addWeighted(np.full_like(img, 255), a, img, 1 - a, 0, img)
        fade = max(0.0, 1 - q / 16)                                        # shockwave ring: grows, thins, fades
        if fade > 0:
            ring = int(R * (0.9 + 2.6 * ease_out(min(1, q / 14))))
            th = max(1, int(9 * (1 - min(1, q / 14))))
            layer = img.copy()
            cv2.circle(layer, tuple(np.int32(self.c)), ring, MAGENTA, th, cv2.LINE_AA)
            cv2.addWeighted(layer, fade, img, 1 - fade, 0, img)
        head_k = ease_out(min(1, q / 10))                                  # rays: head shoots out, tail catches up
        tail_k = ease_out(max(0.0, min(1, (q - 4) / 14))) * 0.98
        for r in self.rays:
            head, tail = R * 0.55 + r["L"] * head_k, R * 0.55 + r["L"] * tail_k
            if head - tail < 2:
                continue
            pts = []
            for d in np.linspace(tail, head, 6):
                ang = r["a"] + r["curve"] * (d - R * 0.55) / (R * 2)      # slight bend = hand-drawn look
                pts.append((self.c[0] + d * math.cos(ang), self.c[1] + d * math.sin(ang)))
            cv2.polylines(img, [np.int32(pts)], False, r["col"], r["w"], cv2.LINE_AA)
        for p in self.parts:                                               # confetti
            if p["r"] > 1:
                cv2.circle(img, tuple(np.int32(p["p"])), int(p["r"]), p["col"], -1, cv2.LINE_AA)


# ------------------------------------------------------------------ hand analysis
def analyse(lm, w, h):
    """returns dict: fingertips, finger-up flags, finger gap, thumb-pinch distance, palm scale."""
    P = lambda i: np.array([lm[i].x * w, lm[i].y * h])
    wrist = P(0)
    def up(tip, pip):                         # orientation-independent "finger extended" test
        return np.linalg.norm(P(tip) - wrist) > np.linalg.norm(P(pip) - wrist) * 1.15
    idx, mid = up(8, 6), up(12, 10)
    scale = np.linalg.norm(P(9) - wrist) + 1e-6                  # palm size for normalising
    gap = np.linalg.norm(P(8) - P(12)) / scale                   # distance index<->middle tip
    pinch = np.linalg.norm(P(4) - P(DOT_TIP)) / scale            # thumb tip <-> finger tip (0 = touching)
    return dict(i=P(8), m=P(12), idx=idx, mid=mid, apart=gap > 0.45, scale=scale,
                wrist=wrist, dot=P(DOT_TIP), pinch=pinch, gap=gap)


# ------------------------------------------------------------------ main effect
class Effect:
    RED_ONLY, VS, MERGE, MAGENTA, DOT, BOOM, GONE = range(7)

    def __init__(self):
        self.state, self.t = self.GONE, 0
        self.red, self.blue, self.mag = Blob(RED), Blob(BLUE, dot=True), Blob(MAGENTA)
        self.strokes, self.miss = [], 0
        self.boom, self.cool = None, 0                 # active explosion, cooldown after it
        self.close_n = 0                               # consecutive frames the two fingers are together
        self.flick_n = 0                               # consecutive frames the flick condition held
        self.prev_rel, self.pinch_hist = None, deque(maxlen=8)   # history for flick detection

    def flicked(self, info):
        """True when the finger is tossed: released from the thumb, or moves fast relative to the wrist."""
        rel = info["dot"] - info["wrist"]                         # fingertip position relative to wrist
        speed = 0.0 if self.prev_rel is None else np.linalg.norm(rel - self.prev_rel) / info["scale"]
        self.prev_rel = rel
        self.pinch_hist.append(info["pinch"])
        released = len(self.pinch_hist) >= 4 and min(self.pinch_hist) < FLICK_CLOSED and info["pinch"] > FLICK_OPEN
        raw = released or speed > FLICK_SPEED
        self.flick_n = self.flick_n + 1 if raw else 0             # must persist: ignores 1-2 frame tracker glitches
        return self.flick_n >= FLICK_FRAMES

    def boom_now(self, info):
        """Start the explosion at the dot (also called by the 'b' key for testing)."""
        if self.state == self.BOOM:
            return
        c = self.mag.pos.p if self.state == self.DOT else (info["dot"] if info else self.mag.pos.p)
        R = (info["scale"] if info else 200) * 0.9
        self.boom = Boom(c, R)
        self.goto(self.BOOM)

    def shake(self):
        return self.boom.shake() if self.state == self.BOOM else (0, 0)

    def goto(self, s):
        self.state, self.t = s, 0

    def update(self, info, w, h):
        self.t += 1
        self.cool = max(0, self.cool - 1)
        if info is None:                              # hand lost for a while -> fade everything
            self.miss += 1
            self.prev_rel = None; self.pinch_hist.clear(); self.close_n = 0; self.flick_n = 0   # tracking gap: forget flick history
            limit = {self.MAGENTA: 90, self.DOT: 45}.get(self.state, 10)   # motion blur can drop the hand briefly
            if self.miss > limit and self.state not in (self.GONE, self.BOOM):
                self.goto(self.GONE)
            info_ok = False
        else:
            self.miss, info_ok = 0, True

        up_off = lambda b: np.array([0, -b.r * 1.3 - 6])          # blobs float above the fingertip
        if info_ok:
            i, m = info["i"], info["m"]
            two_up, one_up = info["idx"] and info["mid"], info["idx"] and not info["mid"]

            flick = self.flicked(info)                                       # update flick history every frame
            self.close_n = self.close_n + 1 if info["gap"] < 0.30 else 0      # fingers together for N frames
            if self.state == self.GONE and one_up and self.cool == 0:
                self.goto(self.RED_ONLY)
                self.red.pos = Spring(i + [0, -30]); self.red.r = 0.0
            if self.state == self.RED_ONLY and two_up and info["apart"]:   # (DOT is sticky: only a flick ends it)
                self.goto(self.VS)
                self.blue.pos = Spring(m + [30, -30]); self.blue.r = 0.0
            if self.state == self.VS and two_up and self.close_n >= 3 and self.t > 15:   # ignore finger jitter
                self.goto(self.MERGE)                                       # fingers snap together
                self.strokes += [Stroke(((self.red.pos.p + self.blue.pos.p) / 2), burst=True, life=6) for _ in range(14)]
            if self.state == self.MAGENTA and not two_up and self.t > 45:    # V closed -> small dot appears
                self.goto(self.DOT)
                self.mag.pos = Spring(info["dot"])
            if self.state == self.DOT and self.t > 12 and flick:             # toss the finger -> BOOM
                self.boom_now(info)

        R = info["scale"] * 0.32 if info_ok else 20
        if self.state == self.RED_ONLY and info_ok:
            self.red.update(info["i"] + up_off(self.red), R * ease_out(self.t / 15))
        elif self.state == self.VS and info_ok:
            self.red.update(info["i"] + up_off(self.red), R)
            self.blue.update(info["m"] + up_off(self.blue) + [30, 0], R * 0.85 * ease_out(self.t / 12))
        elif self.state == self.MERGE:
            mid = (self.red.pos.p + self.blue.pos.p) / 2
            k = min(self.t / 10, 1)
            self.red.update(lerp(self.red.pos.p, mid, .35), R * (1 - .3 * k))   # rush together
            self.blue.update(lerp(self.blue.pos.p, mid, .35), R * (1 - .3 * k))
            self.red.colour = lerp_col(RED, MAGENTA, k); self.blue.colour = lerp_col(BLUE, MAGENTA, k)
            if self.t > 10:                                                     # blur-fade into magenta
                self.mag.pos = Spring(mid); self.mag.r = R * 0.5
                self.mag.blur, self.mag.alpha = 9, 0.0
                self.goto(self.MAGENTA)
                self.red.colour, self.blue.colour = RED, BLUE
                self.red.r = self.blue.r = 0
        elif self.state == self.MAGENTA and info_ok:
            tgt = (info["i"] + info["m"]) / 2 + up_off(self.mag)
            self.mag.update(tgt, R * 1.2)
            self.mag.alpha = min(1, self.mag.alpha + 0.12)                     # fade in from blur
            self.mag.blur = max(0, self.mag.blur - 1)
            if np.linalg.norm(self.mag.pos.v) > 14 and self.t % 2 == 0:        # fast move -> scribbles
                self.strokes.append(Stroke(self.mag.pos.p))
        elif self.state == self.MAGENTA and not info_ok:
            self.mag.pos.v *= 0.8                                              # hand lost: blob settles
        elif self.state == self.DOT and info_ok:
            self.mag.update(info["dot"], 6)
        elif self.state == self.BOOM:
            self.boom.update()
            if self.boom.done:
                self.boom, self.cool = None, 45                              # short pause before the loop can restart
                self.goto(self.GONE); self.mag.r = 0
        elif self.state == self.GONE:
            for b in (self.red, self.blue, self.mag):
                b.r = lerp(b.r, 0, 0.3)

    def draw(self, img):
        if self.state in (self.RED_ONLY, self.VS, self.MERGE):
            self.red.draw(img)
        if self.state in (self.VS, self.MERGE):
            self.blue.draw(img)
        if self.state in (self.MAGENTA, self.DOT, self.GONE):
            self.mag.draw(img)
        if self.state == self.BOOM and self.boom:
            self.boom.draw(img)
        self.strokes = [s for s in self.strokes if s.draw(img)]


# ------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0")          # "0" = webcam, or path to a video file
    ap.add_argument("--out", default="")           # optional: save result to this .mp4
    ap.add_argument("--no-show", action="store_true")   # skip the preview window
    a = ap.parse_args()

    cap = cv2.VideoCapture(int(a.src) if a.src.isdigit() else a.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if a.out else None
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    fx = Effect()
    is_file = not a.src.isdigit()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not is_file:
            frame = cv2.flip(frame, 1)                                   # mirror webcam
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))      # find the hand
        info = analyse(res.multi_hand_landmarks[0].landmark, W, H) if res.multi_hand_landmarks else None
        fx.update(info, W, H)                                            # advance animation
        fx.draw(frame)                                                   # draw blobs on frame
        dx, dy = fx.shake()                                              # camera shake right after the boom pops
        if dx or dy:
            frame = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
        if out: out.write(frame)
        if not a.no_show:
            cv2.imshow("blobfx (q quit, b boom)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):                                          # manual boom for testing
                fx.boom_now(info)

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()