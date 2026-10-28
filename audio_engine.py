"""Funciones reproducibles para una app de Señales y Sistemas.

Los filtros no equivalen a separar instrumentos. No descarga audio comercial.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

# Bandas no superpuestas para comparar energías.
BANDS = {
    "Subgraves (20-80 Hz)": (20.0, 80.0),
    "Graves (80-250 Hz)": (80.0, 250.0),
    "Medios (250-2000 Hz)": (250.0, 2000.0),
    "Medios-altos (2-4 kHz)": (2000.0, 4000.0),
    "Agudos (4-12 kHz)": (4000.0, 12000.0),
}
BAND_COLORS = ["#4B91E2", "#14B8A6", "#FCB64F", "#A58BFF", "#F27E91"]


def ensure_channels(x: np.ndarray) -> np.ndarray:
    """Audio a (canales, muestras); máximo dos canales."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError("Se requiere audio mono o estéreo.")
    if a.shape[0] > 2:
        raise ValueError("El número de canales no es compatible.")
    return a


def mono(x: np.ndarray) -> np.ndarray:
    return ensure_channels(x).mean(axis=0)


def synthetic_demo(fs: int = 24000, duration: float = 12.0) -> tuple[np.ndarray, int]:
    """Ejemplo sintetizado para probar interfaz; NO es una grabación musical."""
    t = np.arange(int(fs * duration), dtype=float) / fs
    rng = np.random.default_rng(32)
    wobble = .58 + .34 * np.sin(2 * np.pi * 1.8 * t)
    subbass = .40 * wobble * np.sin(2 * np.pi * 52 * t)
    low = .13 * np.sin(2 * np.pi * 104 * t)
    mid = .12 * np.sin(2 * np.pi * (415 * t + 3 * t**2))
    high = .07 * np.sin(2 * np.pi * 5100 * t)
    # Pulsos con ataque rápido de percusión SINTÉTICA.
    hits = np.zeros_like(t)
    for beat in np.arange(.3, duration, .6):
        z = t - beat
        idx = (z >= 0) & (z < .11)
        hits[idx] += .25 * np.exp(-36 * z[idx]) * np.sin(2*np.pi*150*z[idx])
    noise = .009 * rng.normal(size=t.size)
    x = subbass + low + mid + high + hits + noise
    return np.stack([x, .92*x]).astype(np.float32), fs


def band_limits(sr: int, lower: float, upper: float) -> tuple[float,float] | None:
    upper = min(float(upper), .97 * sr / 2)
    lower = float(lower)
    if lower <= 0 or upper <= lower or upper <= 1:
        return None
    return lower, upper


def filter_audio(x: np.ndarray, sr: int, kind: str,
                 fc: float = 120.0, lower: float = 80.0,
                 upper: float = 250.0, order: int = 4) -> np.ndarray:
    """Filtro IIR bidireccional por secciones de segundo orden.

    Fase neta cero (salvo bordes), amplitud efectiva |H|², no causal en vivo.
    """
    x = ensure_channels(x)
    nyq = sr / 2
    if kind == "Pasa-bajas":
        if not 0 < fc < .98*nyq:
            raise ValueError("La frecuencia de corte debe ser inferior a Nyquist.")
        sos = signal.butter(order, fc, fs=sr, btype="lowpass", output="sos")
    elif kind == "Pasa-altas":
        if not 0 < fc < .98*nyq:
            raise ValueError("La frecuencia de corte debe ser inferior a Nyquist.")
        sos = signal.butter(order, fc, fs=sr, btype="highpass", output="sos")
    elif kind == "Pasa-banda":
        limits = band_limits(sr, lower, upper)
        if limits is None:
            raise ValueError("La banda seleccionada está fuera del rango de Nyquist.")
        sos = signal.butter(order, limits, fs=sr, btype="bandpass", output="sos")
    else:
        raise ValueError("Tipo de filtro desconocido.")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def filter_band(x: np.ndarray, sr: int, lower: float, upper: float,
                order: int = 4) -> np.ndarray | None:
    limits = band_limits(sr, lower, upper)
    if limits is None:
        return None
    return filter_audio(x, sr, "Pasa-banda", lower=limits[0], upper=limits[1], order=order)


def audio_wav_bytes(x: np.ndarray, sr: int, gain: float = 1.0) -> bytes:
    """WAV PCM16 en memoria sin normalización independiente de cada pista."""
    x = ensure_channels(x)
    a = np.clip(x.T * gain, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class Spectrogram:
    t: np.ndarray
    f: np.ndarray
    magnitude: np.ndarray
    sr: int
    n_fft: int
    hop: int


def spectrogram(x: np.ndarray, sr: int, target_sr: int = 24000,
                n_fft: int = 4096, hop: int = 512) -> Spectrogram:
    y = mono(x)
    effective_sr = min(sr, target_sr)
    if sr != effective_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=effective_sr)
    if y.size < 128:
        raise ValueError("El fragmento es demasiado corto.")
    n_fft = min(n_fft, 2 ** int(np.floor(np.log2(y.size))))
    hop = min(hop, n_fft//4)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True))
    return Spectrogram(
        t=librosa.frames_to_time(np.arange(S.shape[1]), sr=effective_sr, hop_length=hop),
        f=librosa.fft_frequencies(sr=effective_sr, n_fft=n_fft),
        magnitude=S,
        sr=effective_sr,
        n_fft=n_fft,
        hop=hop,
    )


def amplitude_db(magnitude: np.ndarray, reference: float) -> np.ndarray:
    return 20*np.log10(np.maximum(magnitude,1e-12)/max(reference,1e-12))


def energy_evolution(spec: Spectrogram) -> dict[str,np.ndarray]:
    """Proporción de energía STFT para bandas fijas sin solapamiento."""
    power = spec.magnitude**2
    per_band: dict[str, np.ndarray] = {}
    for name, (lo,hi) in BANDS.items():
        idx = (spec.f >= lo) & (spec.f < min(hi,spec.sr/2))
        per_band[name] = power[idx].sum(axis=0) if idx.any() else np.zeros(power.shape[1])
    total = np.maximum(np.sum(list(per_band.values()), axis=0), 1e-18)
    return {name: (value/total)*100 for name,value in per_band.items()}


def downsample_heatmap(spec: Spectrogram, reference: float,
                       fmax: float = 300.0, offset: float = 0,
                       time_cap: int = 700, freq_cap: int = 650):
    index = np.where(spec.f <= fmax)[0]
    if len(index) == 0:
        index = np.array([0])
    if len(index) > freq_cap:
        index = index[np.linspace(0,len(index)-1,freq_cap).astype(int)]
    jt = np.arange(spec.t.size)
    if jt.size > time_cap:
        jt = jt[np.linspace(0,jt.size-1,time_cap).astype(int)]
    return spec.t[jt]+offset, spec.f[index], amplitude_db(spec.magnitude[np.ix_(index,jt)], reference)


def onset_and_tempo(x: np.ndarray, sr: int):
    """Medidas aproximadas: un onset no equivale necesariamente a percusión."""
    y = mono(x)
    if sr > 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    hop = 512
    events = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time")
    try:
        bpm = librosa.feature.tempo(y=y, sr=sr, hop_length=hop)
        estimated = float(np.asarray(bpm).flat[0])
    except Exception:
        estimated = float("nan")
    return np.asarray(events), estimated
