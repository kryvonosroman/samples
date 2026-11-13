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
