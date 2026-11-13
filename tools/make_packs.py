"""Builds the Rap Studio sound packs. Every sound is synthesized from scratch here (no samples)."""
import json, os, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100
OUT = sys.argv[1] if len(sys.argv) > 1 else "packs-out"
rng = np.random.default_rng(7)


# ---------- building blocks ----------
def t_axis(sec): return np.arange(int(sec * SR)) / SR
def env_exp(t, tau): return np.exp(-t / tau)
def attack(t, a): return np.minimum(1.0, t / max(a, 1e-6))
def noise(n): return rng.uniform(-1, 1, n)

def filt(x, kind, f, order=2):
    if isinstance(f, (list, tuple)):
        sos = butter(order, [f[0] / (SR / 2), f[1] / (SR / 2)], btype=kind, output="sos")
    else:
        sos = butter(order, f / (SR / 2), btype=kind, output="sos")
    return sosfilt(sos, x)

def sat(x, drive):  # soft saturation
    return np.tanh(x * drive) / np.tanh(drive)

def sine_sweep(t, f_start, f_end, tau):
    f = f_end + (f_start - f_end) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)

def glide(t, f_from, f_to, start, length):
    k = np.clip((t - start) / length, 0, 1)
    k = k * k * (3 - 2 * k)  # smooth
    f = f_from + (f_to - f_from) * k
    return np.sin(2 * np.pi * np.cumsum(f) / SR)

def finish(x, peak=0.9):
    x = np.asarray(x, dtype=np.float64)
    fade = min(len(x), int(0.005 * SR))
    x[-fade:] *= np.linspace(1, 0, fade)
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x

def note(n):  # MIDI note -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)


# ---------- drums ----------
def kick(punch=170, base=48, tau_pitch=0.03, decay=0.25, click=0.4, drive=1.6, length=0.6, tone_lp=None):
    t = t_axis(length)
    body = sine_sweep(t, punch, base, tau_pitch) * env_exp(t, decay)
    c = filt(noise(len(t)), "highpass", 2500) * env_exp(t, 0.002) * click
    x = sat(body, drive) + c
    if tone_lp: x = filt(x, "lowpass", tone_lp)
    return finish(x)

def bass808(n=30, length=1.6, decay=0.9, drive=2.2, punch=2.0, slide_to=None, slide_at=0.35, slide_len=0.18):
    t = t_axis(length)
    f0 = note(n)
    if slide_to is None:
        f = f0 + f0 * punch * np.exp(-t / 0.025)
        tone = np.sin(2 * np.pi * np.cumsum(f) / SR)
    else:
        k = np.clip((t - slide_at) / slide_len, 0, 1); k = k * k * (3 - 2 * k)
        f = f0 + (note(slide_to) - f0) * k + f0 * punch * np.exp(-t / 0.025)
        tone = np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = sat(tone * attack(t, 0.002) * env_exp(t, decay), drive)
    return finish(x)

def snare(body_f=185, tone=0.6, noise_amt=0.8, hp=1800, lp=9000, tau_noise=0.13, length=0.35, drive=1.3, dull=None):
    t = t_axis(length)
    bend = 1 + 0.3 * np.exp(-t / 0.01)
    b = tone * np.sin(2 * np.pi * np.cumsum(body_f * bend) / SR) * env_exp(t, 0.06) \
        + 0.5 * tone * np.sin(2 * np.pi * np.cumsum(body_f * 1.78 * bend) / SR) * env_exp(t, 0.04)
    n = filt(filt(noise(len(t)), "highpass", hp), "lowpass", lp) * env_exp(t, tau_noise) * noise_amt
    x = sat(b + n, drive)
    if dull: x = filt(x, "lowpass", dull, order=4)
    return finish(x)

def clap(length=0.5, centre=1200, tail=0.14, spread=(0.0, 0.011, 0.022, 0.031)):
    t = t_axis(length)
    e = np.zeros_like(t)
    for s in spread:
        e = np.maximum(e, np.where(t >= s, np.exp(-(t - s) / 0.006), 0))
    e = np.maximum(e, np.where(t >= spread[-1], 0.7 * np.exp(-(t - spread[-1]) / tail), 0))
    x = filt(filt(noise(len(t)), "bandpass", (centre * 0.6, centre * 1.8)), "highpass", 500) * e
    return finish(x)

def metal(t, scale=1.7):
    freqs = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * scale
    ph = np.outer(t, freqs) % 1.0
    return np.where(ph < 0.5, 1.0, -1.0).mean(axis=1)

def hat(decay=0.035, length=0.2, hp=7000, dark=None, noise_mix=0.35):
    t = t_axis(length)
    x = filt(filt(metal(t), "highpass", hp), "highpass", hp) + noise_mix * filt(noise(len(t)), "highpass", hp + 1000)
    x = x * attack(t, 0.001) * env_exp(t, decay)
    if dark: x = filt(x, "lowpass", dark)
    return finish(x)

def shaker(length=0.18, centre=5000, tau=0.045, grit=0.0):
    t = t_axis(length)
    e = attack(t, 0.012) * env_exp(t, tau)
    n = noise(len(t))
    if grit > 0:  # "sandpaper": crackly bursts instead of smooth noise
        mask = (rng.uniform(0, 1, len(t)) < grit).astype(float)
        n = n * (0.35 + mask * 2.0)
    x = filt(n, "bandpass", (centre * 0.6, centre * 1.6)) * e
    return finish(x)

def rim(length=0.12, f1=1700, f2=480):
    t = t_axis(length)
    x = 0.6 * np.sin(2 * np.pi * f1 * t) * env_exp(t, 0.012) + 0.4 * np.sin(2 * np.pi * f2 * t) * env_exp(t, 0.02) \
        + 0.5 * filt(noise(len(t)), "bandpass", (2000, 4500)) * env_exp(t, 0.004)
    return finish(x)

def tom(f=110, length=0.5):
    t = t_axis(length)
    x = sine_sweep(t, f * 1.6, f, 0.04) * env_exp(t, 0.18) + 0.2 * filt(noise(len(t)), "lowpass", 3000) * env_exp(t, 0.01)
    return finish(sat(x, 1.4))


# ---------- melodic one-shots ----------
def fm_bell(n=72, length=2.0, ratio=3.5, index=3.0, tau=0.7, dark=None):
    t = t_axis(length)
    f = note(n)
    mod = index * np.exp(-t / 0.4) * np.sin(2 * np.pi * f * ratio * t)
    x = np.sin(2 * np.pi * f * t + mod) * attack(t, 0.002) * env_exp(t, tau)
    x += 0.3 * np.sin(2 * np.pi * f * 2.01 * t) * env_exp(t, tau * 0.5)
    if dark: x = filt(x, "lowpass", dark)
    return finish(x)

def pluck(n=60, length=0.9, tau=0.25, bright=2500):
    # Karplus-Strong string
    f = note(n); period = int(SR / f)
    buf = noise(period); out = np.zeros(int(length * SR))
    for i in range(len(out)):
        out[i] = buf[i % period]
        buf[i % period] = 0.5 * (buf[i % period] + buf[(i + 1) % period]) * 0.996
    t = t_axis(length)
    return finish(filt(out, "lowpass", bright) * env_exp(t, tau * 2))

def piano_stab(notes=(60, 63, 67), length=1.2):
    t = t_axis(length)
    x = np.zeros_like(t)
    for n in notes:
        f = note(n)
        for h, a in enumerate([1.0, 0.5, 0.3, 0.2, 0.12, 0.08], start=1):
            x += a / len(notes) * np.sin(2 * np.pi * f * h * (1 + 0.0004 * h * h) * t) * env_exp(t, 0.6 / h ** 0.6)
    x = x * attack(t, 0.003) + 0.05 * filt(noise(len(t)), "bandpass", (1500, 5000)) * env_exp(t, 0.01)
    return finish(sat(x, 1.2))

def pad_stab(notes=(57, 60, 64), length=1.6, cutoff=1800):
    t = t_axis(length)
    x = np.zeros_like(t)
    for n in notes:
        for det in (-0.12, 0.0, 0.11):
            f = note(n + det)
            x += (2 * ((f * t) % 1.0) - 1)  # saw
    x = filt(x, "lowpass", cutoff, order=4) * attack(t, 0.04) * env_exp(t, 0.5)
    return finish(x)

def brass_stab(notes=(48, 55, 60), length=0.7):
    t = t_axis(length)
    x = np.zeros_like(t)
    for n in notes:
        f = note(n)
        x += 2 * ((f * t) % 1.0) - 1
    sweep = 400 + 2600 * np.exp(-t / 0.08)
    out = np.zeros_like(x); y = 0.0
    for i in range(len(x)):  # one-pole lowpass with moving cutoff
        a = np.exp(-2 * np.pi * sweep[i] / SR); y = (1 - a) * x[i] + a * y; out[i] = y
    return finish(sat(out * attack(t, 0.01) * env_exp(t, 0.25), 1.5))


# ---------- FX (original "spell" sounds, nothing taken from any game) ----------
def fx_spell_cast(length=1.4):
    t = t_axis(length)
    x = np.zeros_like(t)
    for i, ratio in enumerate([1.0, 1.5, 2.0, 2.52, 3.0]):
        f = 300 * ratio * (1 + 2.5 * t / length)
        x += np.sin(2 * np.pi * np.cumsum(f) / SR + i) * (0.6 ** i)
    shimmer = filt(noise(len(t)), "highpass", 6000) * 0.25
    e = attack(t, length * 0.8) * np.exp(-np.maximum(0, t - length * 0.8) / 0.08)
    return finish((x + shimmer) * e)

def fx_blink(length=0.35):
    t = t_axis(length)
    f = 3000 * np.exp(-t / 0.05) + 200
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(t, 0.08) + 0.3 * filt(noise(len(t)), "highpass", 4000) * env_exp(t, 0.02)
    return finish(x)

def fx_arcane_hit(length=1.6):
    t = t_axis(length)
    boom = sine_sweep(t, 120, 40, 0.08) * env_exp(t, 0.35)
    ring = sum(np.sin(2 * np.pi * f * t) * env_exp(t, 0.6) * a for f, a in [(523, 0.4), (787, 0.3), (1213, 0.2), (1649, 0.15)])
    crack = filt(noise(len(t)), "highpass", 2000) * env_exp(t, 0.02)
    return finish(sat(boom, 2.0) + ring * 0.5 + crack * 0.6)

def fx_whoosh(length=0.9):
    t = t_axis(length)
    e = np.sin(np.pi * np.clip(t / length, 0, 1)) ** 2
    n = noise(len(t)); out = np.zeros_like(n); y = 0.0
    cutoff = 300 + 4000 * np.sin(np.pi * t / length)
    for i in range(len(n)):
        a = np.exp(-2 * np.pi * cutoff[i] / SR); y = (1 - a) * n[i] + a * y; out[i] = y
    return finish(out * e)

def fx_riser(length=2.0):
    t = t_axis(length)
    f = 200 * 2 ** (3 * t / length)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.4 * filt(noise(len(t)), "highpass", 3000)
    return finish(x * (t / length) ** 2)

def fx_impact(length=1.5):
    t = t_axis(length)
    x = sat(sine_sweep(t, 90, 32, 0.15) * env_exp(t, 0.5), 2.5) + 0.5 * filt(noise(len(t)), "lowpass", 2500) * env_exp(t, 0.12)
    return finish(x)

def vinyl_crackle(length=2.0):
    n = int(length * SR)
    x = np.zeros(n)
    pops = rng.integers(0, n, 140)
    x[pops] = rng.uniform(-1, 1, 140)
    x = filt(x, "bandpass", (800, 6000)) * 3 + 0.03 * filt(noise(n), "bandpass", (300, 3000))
    return finish(x, 0.5)


# ---------- phonk ----------
def cowbell(f1=562, f2=845, length=0.45, drive=2.5, tail=0.16, crush_bits=None):
    """Drum-machine cowbell: two square waves, band-passed, a sharp hit and a ring; driven like phonk."""
    t = t_axis(length)
    sq = np.sign(np.sin(2 * np.pi * f1 * t)) + np.sign(np.sin(2 * np.pi * f2 * t))
    tone = filt(filt(sq, "bandpass", (1200, 3800)), "highpass", 450)
    env = attack(t, 0.0008) * (0.65 * env_exp(t, 0.018) + 0.35 * env_exp(t, tail))
    x = sat(tone * env * 2.0, drive)
    if crush_bits:
        q = 2 ** (crush_bits - 1)
        x = np.round(x * q) / q
    return finish(x)

def vox_chop(f0=180, vowel="ah", length=0.45, drop=0.0, grit=1.5):
    """A sung-sounding vocal chop built from a buzzy source and vowel formants (no recorded voice)."""
    formants = {"ah": (800, 1150, 2900), "oh": (450, 800, 2830), "eh": (530, 1840, 2480), "uh": (640, 1190, 2390)}[vowel]
    t = t_axis(length)
    f = f0 * (1 - drop * np.clip(t / length, 0, 1))
    phase = np.cumsum(f) / SR
    source = 2 * (phase % 1.0) - 1  # saw
    vib = 1 + 0.01 * np.sin(2 * np.pi * 5.5 * t)
    source = source * vib
    x = sum(filt(source, "bandpass", (fc * 0.85, fc * 1.15)) * g for fc, g in zip(formants, (1.0, 0.6, 0.25)))
    env = attack(t, 0.01) * np.clip((length - t) / 0.06, 0, 1)
    return finish(sat(x * env * 3, grit))

def reverse_crash(length=1.6):
    t = t_axis(length)
    x = filt(noise(len(t)) + 0.5 * metal(t, 2.3), "highpass", 4000) * env_exp(t, 0.5)
    return finish(x[::-1])



# ---------- zxc / memphis ----------
def reverb(x, seconds=1.8, mix=0.3, dark=6000, seed=5):
    """A simple room: convolve with decaying, filtered noise (no recorded impulse)."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = r.uniform(-1, 1, n) * np.exp(-t / (seconds / 5.0))
    ir = filt(ir, "lowpass", dark)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    wet = np.convolve(x, ir)[: len(x) + n // 2]
    dry = np.concatenate([x, np.zeros(len(wet) - len(x))])
    return dry * (1 - mix) + wet * mix * 0.6

def chorus(x, depth_ms=3.0, rate=0.6):
    n = len(x)
    t = np.arange(n) / SR
    d = (depth_ms * 0.001 * SR) * (1 + np.sin(2 * np.pi * rate * t)) * 0.5 + 0.01 * SR
    idx = np.clip(np.arange(n) - d, 0, n - 1)
    i0 = idx.astype(int); fr = idx - i0
    i1 = np.minimum(i0 + 1, n - 1)
    return 0.7 * x + 0.5 * (x[i0] * (1 - fr) + x[i1] * fr)

def string_pluck(freq, length, decay=0.996, bright=3500, seed=0):
    """Karplus-Strong guitar string."""
    r = np.random.default_rng(seed)
    n = int(length * SR)
    period = max(2, int(round(SR / freq)))
    line = filt(r.uniform(-1, 1, period * 4), "lowpass", bright)[-period:]
    out = np.zeros(n)
    last = 0.0
    buf = list(line)
    for i in range(n):
        k = i % period
        y = buf[k]
        buf[k] = decay * 0.5 * (y + last)
        last = y
        out[i] = y
    return out

def guitar_loop(notes, bpm=140, bars=2, step=2, length_each=1.2, seed=1, lofi=True):
    """An arpeggio loop: one note every `step` 16ths (MIDI notes), chorus, room, tape dulling."""
    sixteenth = 60.0 / bpm / 4
    total = int(bars * 16 * sixteenth * SR)
    out = np.zeros(total + int(length_each * SR))
    for i, n in enumerate(notes):
        start = int(i * step * sixteenth * SR)
        tone = string_pluck(note(n), length_each, seed=seed + i) * (0.9 if i % 4 == 0 else 0.7)
        out[start:start + len(tone)] += tone
    out = chorus(out)
    out = reverb(out, 2.2, 0.35)
    if lofi:
        out = filt(out, "lowpass", 5200)
        t = np.arange(len(out)) / SR
        out *= 1 + 0.03 * np.sin(2 * np.pi * 0.5 * t)  # tape wobble in level
    return finish(out[:total])

def pad_loop(chords, bpm=140, bars_each=1, cutoff=1600, seed=2):
    """Hazy saw pad, one chord per bar, lo-fi."""
    bar = 60.0 / bpm * 4
    n_each = int(bar * bars_each * SR)
    out = []
    for c in chords:
        t = np.arange(n_each) / SR
        x = np.zeros(n_each)
        for m in c:
            for det in (0.997, 1.003):
                ph = (note(m) * det * t) % 1.0
                x += 2 * ph - 1
        env = np.minimum(1, t / 0.25) * np.minimum(1, (t[-1] - t + 1e-9) / 0.2)
        out.append(x * env / (2 * len(c)))
    x = filt(np.concatenate(out), "lowpass", cutoff)
    x = reverb(x, 2.5, 0.4, dark=3000)
    x = x + 0.01 * filt(noise(len(x)), "bandpass", (1000, 6000))  # tape hiss
    return finish(x[: n_each * len(chords)])

def vox_ooh(f0=220, length=1.2, vowel="oh", wet=0.45, seed=3):
    x = vox_chop(f0=f0, vowel=vowel, length=length, grit=1.2)
    x = reverb(x, 2.5, wet, dark=5000, seed=seed)
    return finish(x)

def tape_stop(length=1.2, f=110):
    t = t_axis(length)
    speed = np.clip(1 - t / length, 0, 1) ** 1.5
    phase = np.cumsum(f * speed) / SR
    x = np.sign(np.sin(2 * np.pi * phase)) * 0.4 + np.sin(2 * np.pi * phase * 0.5)
    return finish(filt(x, "lowpass", 1500) * np.minimum(1, (length - t) / 0.05))

def dusty(x, bits=8, cutoff=7000, drive=2.0):
    x = sat(np.asarray(x), drive)
    q = 2 ** (bits - 1)
    x = np.round(x * q) / q
    x = filt(x, "lowpass", cutoff)
    return finish(x)

def memphis_cowbell_loop(bpm=130, seed=4):
    sixteenth = 60.0 / bpm / 4
    pattern = [(0, 0), (3, 0), (6, 3), (8, 0), (10, 7), (12, 5), (14, 3),
               (16, 0), (19, 0), (22, 3), (24, 8), (26, 7), (28, 5), (30, 3)]
    total = int(32 * sixteenth * SR)
    out = np.zeros(total + SR)
    base = cowbell(drive=4.0, tail=0.2)
    for step, semi in pattern:
        ratio = 2 ** (semi / 12)
        idx = np.arange(0, len(base) - 1, ratio)
        shifted = np.interp(idx, np.arange(len(base)), base)
        s0 = int(step * sixteenth * SR)
        out[s0:s0 + len(shifted)] += shifted * 0.6
    out = reverb(out, 1.2, 0.2)
    return dusty(out[:total], bits=9, cutoff=8000, drive=1.5)

# ---------- the packs ----------
PACKS = [
    {
        "id": "dota-rap-kit", "name": "Dota Rap Kit", "style": "Dota rap / drill-adjacent", "tempo": "140-150 BPM",
        "description": "Grimy, drill-adjacent kit for Dota rap: bulbous 808s, flat kicks, dull papery snares, "
                       "sandpaper percussion, dark bells and original spell FX.",
        "sounds": [
            ("Kick Flat", "Kick", lambda: kick(punch=150, base=55, tau_pitch=0.02, decay=0.14, click=0.25, drive=1.2)),
            ("Kick Thump", "Kick", lambda: kick(punch=190, base=45, decay=0.3, drive=2.2)),
            ("808 Bulb", "808", lambda: bass808(n=30, decay=1.1, drive=3.0, punch=2.5)),
            ("808 Slide Up", "808", lambda: bass808(n=30, slide_to=37, decay=1.0, drive=2.6)),
            ("808 Slide Down", "808", lambda: bass808(n=37, slide_to=30, decay=1.0, drive=2.6)),
            ("808 Grit", "808", lambda: bass808(n=33, decay=0.8, drive=5.0)),
            ("Snare Dull", "Snare", lambda: snare(body_f=170, noise_amt=0.6, tau_noise=0.1, dull=3500)),
            ("Snare Paper", "Snare", lambda: snare(body_f=210, tone=0.35, noise_amt=1.0, hp=2500, lp=7000, tau_noise=0.09, dull=5500)),
            ("Rim Dry", "Perc", lambda: rim(f1=1500, f2=420)),
            ("Sand Shaker", "Perc", lambda: shaker(centre=4200, tau=0.05, grit=0.02)),
            ("Grit Perc", "Perc", lambda: shaker(length=0.12, centre=2500, tau=0.025, grit=0.05)),
            ("Hat Dark", "Hi-Hat", lambda: hat(decay=0.03, dark=9000)),
            ("Open Hat Dark", "Hi-Hat", lambda: hat(decay=0.25, length=0.8, dark=8500)),
            ("Bell Dark", "Melodic", lambda: fm_bell(n=69, ratio=2.4, index=2.2, tau=0.9, dark=4000)),
            ("Pluck Dead", "Melodic", lambda: pluck(n=57, bright=1800)),
            ("FX Spell Cast", "FX", fx_spell_cast),
            ("FX Blink", "FX", fx_blink),
            ("FX Arcane Hit", "FX", fx_arcane_hit),
            ("FX Whoosh", "FX", fx_whoosh),
        ],
    },
    {
        "id": "drill-kit", "name": "Drill Kit", "style": "UK / NY drill", "tempo": "140-145 BPM",
        "description": "Sliding 808s, crisp drill snares, bouncy hats and percs.",
        "sounds": [
            ("Kick Punch", "Kick", lambda: kick(punch=180, base=50, decay=0.2, click=0.35)),
            ("808 Slide A", "808", lambda: bass808(n=33, slide_to=40, slide_at=0.25)),
            ("808 Slide B", "808", lambda: bass808(n=40, slide_to=33, slide_at=0.25)),
            ("808 Long", "808", lambda: bass808(n=31, length=2.2, decay=1.4)),
            ("Snare Drill", "Snare", lambda: snare(body_f=230, tone=0.5, hp=2200, tau_noise=0.11)),
            ("Clap Tight", "Clap", lambda: clap(tail=0.09)),
            ("Hat Tick", "Hi-Hat", lambda: hat(decay=0.02, length=0.12)),
            ("Open Hat", "Hi-Hat", lambda: hat(decay=0.2, length=0.7)),
            ("Perc Wood", "Perc", lambda: tom(f=420, length=0.18)),
            ("Shaker", "Perc", lambda: shaker()),
        ],
    },
    {
        "id": "detroit-kit", "name": "Detroit Kit", "style": "Detroit rap", "tempo": "90-100 BPM",
        "description": "Knocking kicks, hard claps, fast hats and piano stabs.",
        "sounds": [
            ("Kick Knock", "Kick", lambda: kick(punch=220, base=52, decay=0.18, click=0.6, drive=3.0)),
            ("Snare Hard", "Snare", lambda: snare(body_f=200, tone=0.7, drive=2.2)),
            ("Clap Wide", "Clap", lambda: clap(tail=0.18, spread=(0.0, 0.009, 0.019, 0.03, 0.042))),
            ("Hat Fast", "Hi-Hat", lambda: hat(decay=0.018, length=0.1)),
            ("808 Short", "808", lambda: bass808(n=34, length=0.7, decay=0.3, drive=3.5)),
            ("Piano Stab Minor", "Melodic", lambda: piano_stab((60, 63, 67))),
            ("Piano Stab Dark", "Melodic", lambda: piano_stab((55, 58, 62, 65))),
            ("Brass Stab", "Melodic", brass_stab),
        ],
    },
    {
        "id": "boom-bap-kit", "name": "Boom Bap Kit", "style": "Boom bap", "tempo": "85-95 BPM",
        "description": "Dusty kicks, cracking snares, vinyl hats and crackle.",
        "sounds": [
            ("Kick Dusty", "Kick", lambda: kick(punch=140, base=60, decay=0.22, click=0.2, tone_lp=2500)),
            ("Snare Crack", "Snare", lambda: snare(body_f=190, tone=0.5, noise_amt=1.0, tau_noise=0.16, lp=7000)),
            ("Rim Wood", "Perc", lambda: rim(f1=1200, f2=380)),
            ("Hat Vinyl", "Hi-Hat", lambda: hat(decay=0.04, dark=7000, noise_mix=0.6)),
            ("Shaker Soft", "Perc", lambda: shaker(centre=3500, tau=0.06)),
            ("Pad Soul", "Melodic", lambda: pad_stab((57, 60, 64, 67))),
            ("Vinyl Crackle", "FX", vinyl_crackle),
            ("Tom Low", "Perc", lambda: tom(f=95)),
        ],
    },
    {
        "id": "phonk-kit", "name": "Phonk Kit", "style": "Phonk / drift phonk", "tempo": "130-160 BPM",
        "description": "Driven cowbells to play melodies with, clipped and sliding 808s, hard kicks, "
                       "original synthetic vocal chops and a reverse crash.",
        "sounds": [
            ("Cowbell Classic", "Melodic", lambda: cowbell()),
            ("Cowbell Dirty", "Melodic", lambda: cowbell(drive=6.0, tail=0.22)),
            ("Cowbell Crushed", "Melodic", lambda: cowbell(drive=4.0, crush_bits=5)),
            ("Cowbell Low", "Melodic", lambda: cowbell(f1=420, f2=632, tail=0.2, drive=3.5)),
            ("808 Clipped", "808", lambda: bass808(n=33, decay=1.0, drive=8.0, punch=2.5)),
            ("808 Slide Phonk", "808", lambda: bass808(n=28, slide_to=40, slide_at=0.2, slide_len=0.12, drive=6.0)),
            ("Kick Phonk", "Kick", lambda: kick(punch=200, base=50, decay=0.2, click=0.5, drive=4.0)),
            ("Snare Phonk", "Snare", lambda: snare(body_f=200, tone=0.6, noise_amt=0.9, tau_noise=0.15, drive=2.5)),
            ("Clap Phonk", "Clap", lambda: clap(tail=0.2)),
            ("Hat Phonk", "Hi-Hat", lambda: hat(decay=0.025, length=0.12, hp=8000)),
            ("Vox Chop Ah", "Vocal", lambda: vox_chop(f0=196, vowel="ah")),
            ("Vox Chop Oh", "Vocal", lambda: vox_chop(f0=147, vowel="oh", length=0.6, drop=0.15)),
            ("Vox Chop Hey", "Vocal", lambda: vox_chop(f0=220, vowel="eh", length=0.25, grit=3.0)),
            ("Reverse Crash", "FX", reverse_crash),
        ],
    },
    {
        "id": "zxc-kit", "name": "Zxc Kit", "style": "Zxc / emo-trap", "tempo": "140 BPM",
        "description": "Melancholic guitar arpeggio loops, hazy pads and airy vocal chops in C minor, "
                       "lo-fi drums and soft 808s. Loops are 2 bars at 140 BPM (use Loop to tempo: 8 beats).",
        "sounds": [
            ("Guitar Loop Cm-Ab 140", "Loop", lambda: guitar_loop([60, 63, 67, 72, 75, 72, 67, 63, 56, 60, 63, 68, 72, 68, 63, 60])),
            ("Guitar Loop Eb-Bb 140", "Loop", lambda: guitar_loop([63, 67, 70, 75, 79, 75, 70, 67, 58, 62, 65, 70, 74, 70, 65, 62], seed=20)),
            ("Guitar Loop Slow Cm 140", "Loop", lambda: guitar_loop([60, 67, 63, 70, 60, 67, 63, 72], step=4, length_each=2.0, seed=40)),
            ("Pad Loop Cm-Ab 140", "Loop", lambda: pad_loop([(48, 55, 60, 63), (44, 51, 56, 60)])),
            ("Pad Loop Eb-Bb 140", "Loop", lambda: pad_loop([(51, 58, 63, 67), (46, 53, 58, 62)])),
            ("Guitar Note C", "Melodic", lambda: finish(reverb(chorus(string_pluck(note(60), 2.0, seed=7)), 2.0, 0.3))),
            ("Vocal Ooh C", "Vocal", lambda: vox_ooh(f0=261.6, vowel="oh")),
            ("Vocal Ah G", "Vocal", lambda: vox_ooh(f0=196.0, vowel="ah")),
            ("Vocal Uh Eb", "Vocal", lambda: vox_ooh(f0=155.6, vowel="uh", length=1.6)),
            ("808 Soft C", "808", lambda: bass808(n=36, length=2.0, decay=1.2, drive=1.6, punch=1.5)),
            ("Kick Lo-Fi", "Kick", lambda: dusty(kick(punch=150, base=50, decay=0.25), bits=10, cutoff=6000)),
            ("Snare Lo-Fi", "Snare", lambda: dusty(reverb(snare(body_f=190, tau_noise=0.12), 1.0, 0.25), bits=10, cutoff=7000)),
            ("Hat Soft", "Hi-Hat", lambda: hat(decay=0.03, dark=8000)),
            ("Reverse Guitar Swell", "FX", lambda: finish(guitar_loop([60, 63, 67, 72], step=1, bars=0.5, length_each=2.0, seed=60)[::-1])),
            ("Tape Hiss", "FX", lambda: finish(filt(noise(int(4 * SR)), "bandpass", (800, 7000)), 0.25)),
        ],
    },
    {
        "id": "memphis-phonk-kit", "name": "Memphis Phonk Kit", "style": "Memphis phonk", "tempo": "120-135 BPM",
        "description": "Dusty, dark and slow: bitcrushed drums, distorted cowbell loop and one-shots, eerie bells, "
                       "low synthetic vocal chops and a tape stop. Original sounds.",
        "sounds": [
            ("Cowbell Loop 130", "Loop", lambda: memphis_cowbell_loop()),
            ("Cowbell Dusty", "Melodic", lambda: dusty(cowbell(drive=5.0, tail=0.25), bits=7)),
            ("Bell Eerie", "Melodic", lambda: finish(reverb(fm_bell(n=64, ratio=3.1, index=2.8, tau=1.2, dark=3500), 2.5, 0.35))),
            ("Piano Stab Dark", "Melodic", lambda: dusty(piano_stab((48, 51, 55, 58)), bits=9, cutoff=4500, drive=1.4)),
            ("808 Memphis", "808", lambda: dusty(bass808(n=31, length=1.8, decay=1.0, drive=4.0), bits=10, cutoff=5000, drive=2.5)),
            ("Kick Dusty", "Kick", lambda: dusty(kick(punch=170, base=52, decay=0.2, click=0.5, drive=3.0), bits=8)),
            ("Snare Dusty", "Snare", lambda: dusty(snare(body_f=210, tone=0.6, drive=2.0), bits=7, cutoff=6000)),
            ("Clap Dusty", "Clap", lambda: dusty(clap(tail=0.16), bits=8, cutoff=6500)),
            ("Hat Crunch", "Hi-Hat", lambda: dusty(hat(decay=0.03, length=0.12), bits=6, cutoff=9000)),
            ("Open Hat Crunch", "Hi-Hat", lambda: dusty(hat(decay=0.2, length=0.6), bits=6, cutoff=8000)),
            ("Vox Low Oh", "Vocal", lambda: dusty(vox_chop(f0=98, vowel="oh", length=0.7, drop=0.2, grit=2.5), bits=9, cutoff=5000)),
            ("Vox Low Hey", "Vocal", lambda: dusty(vox_chop(f0=110, vowel="eh", length=0.3, grit=3.5), bits=8, cutoff=5000)),
            ("Tape Stop", "FX", tape_stop),
            ("Vinyl Dust", "FX", vinyl_crackle),
        ],
    },
]


def slug(name): return name.lower().replace(" ", "-")


def main():
    index = {"version": 1, "name": "Rap Studio Sound Store",
             "license": "CC0 1.0: free to use in any music, no credit needed", "packs": []}
    for pack in PACKS:
        folder = os.path.join(OUT, "packs", pack["id"])
        os.makedirs(folder, exist_ok=True)
        entry = {k: pack[k] for k in ("id", "name", "style", "tempo", "description")}
        entry["sounds"] = []
        total = 0
        for name, category, make in pack["sounds"]:
            audio = make()
            path = os.path.join(folder, slug(name) + ".wav")
            wavfile.write(path, SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
            size = os.path.getsize(path)
            total += size
            entry["sounds"].append({"name": name, "category": category,
                                    "file": f"packs/{pack['id']}/{slug(name)}.wav", "bytes": size,
                                    "seconds": round(len(audio) / SR, 2)})
        entry["bytes"] = total
        index["packs"].append(entry)
        print(f"{pack['name']}: {len(entry['sounds'])} sounds, {total / 1024:.0f} KB")
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
