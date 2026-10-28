"""Laboratorio musical interactivo para Señales y Sistemas.

Ejecuta: streamlit run app.py
Audio: subida local o DEMO explícita. Separación de fuentes IA opcional.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from audio_engine import (
    BANDS, BAND_COLORS, amplitude_db, audio_wav_bytes, downsample_heatmap,
    energy_evolution, ensure_channels, filter_audio, filter_band,
    mono, onset_and_tempo, spectrogram, synthetic_demo,
)

st.set_page_config(page_title="Laboratorio Musical | Señales y Sistemas", page_icon="🎧", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 1.45rem; max-width: 1450px; }
  .hero { border: 1px solid #28455c; border-radius: 18px; padding: 20px 24px;
          background: linear-gradient(105deg,#0f2731 5%,#122039 65%,#1c1a39 100%); margin-bottom: 16px; }
  .hero h1 { margin:0; font-size:2rem; color: #e8f8fc; }
  .hero p {color:#b6cbd5; margin:6px 0 0; }
  .subtle {color:#95aebe; font-size:.91rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🎧 Laboratorio musical interactivo</h1>
<p>Escucha, filtra y visualiza cualquier canción con espectrogramas STFT y análisis de fuentes musicales.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def get_duration(path: str) -> float:
    return float(librosa.get_duration(path=path))


@st.cache_data(show_spinner=False, max_entries=12)
def load_clip(path: str, start: float, duration: float) -> tuple[np.ndarray,int]:
    arr, sr = librosa.load(path, sr=None, mono=False, offset=start, duration=duration)
    return ensure_channels(arr), int(sr)


@st.cache_data(show_spinner=False, max_entries=12)
def get_spectrogram(arr: np.ndarray, sr: int, nfft: int):
    return spectrogram(arr, sr, n_fft=nfft, hop=nfft//8)


def create_spectrogram_chart(spec, reference: float, fmax: int, offset: float, title: str) -> go.Figure:
    xs, ys, zs = downsample_heatmap(spec, reference, fmax=fmax, offset=offset)
    fig = go.Figure(go.Heatmap(
        x=xs, y=ys, z=zs, zmin=-85, zmax=0,
        colorscale="Magma", colorbar=dict(title="dB", len=.85),
        hovertemplate="t=%{x:.2f} s<br>f=%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Tiempo en la canción (s)",
        yaxis_title="Frecuencia (Hz)",
        margin=dict(l=8,r=20,t=42,b=20),
        height=365,
    )
    return fig


def create_waveform(x, fs, offset):
    wave = mono(x)
    step = max(1, len(wave)//5000)
    time = offset + np.arange(0,len(wave),step)/fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,y=wave[::step],mode="lines",name="Original",
                             line=dict(color="#37B8DC",width=1.2)))
    fig.update_layout(
        title="Forma de onda del fragmento",xaxis_title="Tiempo en la canción (s)",
        yaxis_title="Amplitud",height=245,margin=dict(l=8,r=12,t=45,b=12),
        showlegend=False,
    )
    return fig


def create_band_energy(spec, start):
    energies = energy_evolution(spec)
    t = spec.t + start
    fig = go.Figure()
    for color,(name,energy) in zip(BAND_COLORS, energies.items()):
        fig.add_trace(go.Scatter(x=t,y=energy,mode="lines",stackgroup="one",name=name,
                                 line=dict(color=color,width=.7)))
    fig.update_layout(title="Energía relativa por banda a lo largo del tiempo",
                      xaxis_title="Tiempo (s)",yaxis_title="Energía relativa (%)",
                      yaxis=dict(range=[0,100]),height=330,
                      margin=dict(l=8,r=12,t=48,b=12),legend=dict(orientation="h",y=-.36))
    return fig


def get_uploaded_path(upload):
    """Guardar en carpeta temporal privada, nunca usar upload.name directamente como ruta."""
    if "session_audio_dir" not in st.session_state:
        st.session_state["session_audio_dir"] = tempfile.mkdtemp(prefix="lab_musical_")
    extension = Path(upload.name).suffix.lower()
    if extension not in {".mp3",".wav",".flac",".ogg",".m4a",".aiff",".aif"}:
        raise ValueError("Formato no admitido. Usa WAV, MP3, FLAC, OGG o M4A.")
    data = upload.getvalue()
    fingerprint = hashlib.sha256(data).hexdigest()[:24]
    dest = Path(st.session_state["session_audio_dir"]) / f"audio_{fingerprint}{extension}"
    if not dest.exists():
        dest.write_bytes(data)
    return dest


def export_bundle(audio, filtered, fs, start, end, metadata):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("original_fragmento.wav", audio_wav_bytes(audio,fs))
        z.writestr("fragmento_filtrado.wav", audio_wav_bytes(filtered,fs))
        z.writestr("resumen.json",json.dumps(metadata,ensure_ascii=False,indent=2))
        z.writestr("LEEME.txt", "Audio analizado localmente. Los filtros no aislan instrumentos. "
                    "Para extraer voz/bateria/guitarra emplea la pestaña 'Fuentes IA' si Demucs esta instalado.\n")
    return buf.getvalue()


def ai_map_data(stems: dict[str,np.ndarray], sr: int):
    """Mapa de energía estimada por fuente, NO clasificación concluyente."""
    if "guitar" in stems:
        guitar = stems["guitar"]
        other = sum((stems[name] for name in ["bass","piano","other"] if name in stems))
        merged = {
            "Voz": stems["vocals"],
            "Cuerdas: guitarra": guitar,
            "Percusión": stems["drums"],
            "Otros (bajo + piano + resto)": other,
        }
    else:
        merged = {
            "Voz": stems["vocals"],
            "Percusión": stems["drums"],
            "Bajo": stems["bass"],
            "Otros (sin separar cuerdas)": stems["other"],
        }
    window = max(1,int(sr*.5))
    n = min(len(a) for a in merged.values())//window
    energies = np.array([
        np.mean(sig[:n*window].reshape(n,window)**2,axis=1) for sig in merged.values()
    ])
    frac=energies/np.maximum(energies.sum(axis=0),1e-12)
    return merged, frac


with st.sidebar:
    st.header("🎵 1. Audio")
    uploaded = st.file_uploader("Sube tu canción", type=["mp3","wav","flac","ogg","m4a","aiff","aif"],
                                help="El audio se procesa en el servidor donde ejecutes esta aplicación.")
    if uploaded is None:
        st.caption("Sin archivo cargado: se utiliza una **demostración sintética** de 12 s.")
    st.divider()

if uploaded is not None:
    try:
        song_path = get_uploaded_path(uploaded)
        total_time = get_duration(str(song_path))
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"No se pudo abrir el archivo: {exc}")
        st.stop()
else:
    song_path = None
    total_time = 12.0
    source_name = "DEMO SINTÉTICA (no es una canción comercial)"

with st.sidebar:
    st.header("✂️ 2. Fragmento")
    if total_time < .5:
        st.error("La canción dura menos de 0.5 s; selecciona otra grabación.")
        st.stop()
    start_default = 55.0 if song_path and "limit" in source_name.lower() and total_time > 80 else 0.0
    end_default = min(total_time,start_default+20.)
    # Garantizar que el estado no contradiga límites de otra canción recién cargada.
    start = st.number_input("Desde el segundo",min_value=0.0,
                            max_value=max(0.,total_time-.5),value=float(start_default),step=.5)
    end_min = min(total_time, start+.5)
    end_default = max(end_min,min(total_time, start+20))
    end = st.number_input("Hasta el segundo",min_value=float(end_min),max_value=float(total_time),
                          value=float(end_default),step=.5)
    if end-start > 45:
        st.warning("Para mantener la interfaz rápida, el fragmento se limita a 45 s.")
        end=start+45
    st.caption(f"Fragmento: {end-start:.1f} s de {total_time:.1f} s")
    st.divider()

try:
    if song_path:
        with st.spinner("Abriendo el fragmento..."):
            audio,fs = load_clip(str(song_path),round(start,2),round(end-start,2))
    else:
        demo,fs=synthetic_demo()
        audio=demo[:,int(start*fs):int(end*fs)]
    if audio.shape[1] < int(.3*fs):
        st.error("Selecciona un fragmento de al menos 0.3 s.")
        st.stop()
except Exception as exc:
    st.error(f"No fue posible decodificar el audio: {exc}")
    st.info("Para algunos formatos MP3/M4A puede ser necesario instalar FFmpeg.")
    st.stop()

with st.sidebar:
    st.header("🎛️ 3. Filtro")
    kind=st.selectbox("Tipo de filtro",["Pasa-bajas","Pasa-altas","Pasa-banda"])
    order=st.select_slider("Orden Butterworth",options=[2,4,6],value=4)
    max_cut=max(60, int(min(5000,fs*.45)))
    if kind == "Pasa-banda":
        lo = st.slider("Frecuencia inferior (Hz)", 20, max_cut-10, min(80,max_cut-20),10)
        hi = st.slider("Frecuencia superior (Hz)",lo+10,max_cut,max(lo+10,min(250,max_cut)),10)
        fc = None
    else:
        fc = st.slider("Frecuencia de corte (Hz)", 20,max_cut,min(120,max_cut),10)
        lo, hi = 20,250
    st.caption("Filtrado hacia adelante y hacia atrás: fase neta cero, no realizable así en tiempo real.")

try:
    with st.spinner("Filtrando..."):
        filtered = filter_audio(audio,fs,kind,fc=fc or 120,lower=lo,upper=hi,order=order)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"**Fuente:** {source_name} · **Segmento:** {start:.1f}–{end:.1f} s · **Muestreo:** {fs:,} Hz".replace(","," "))

col_a,col_b=st.columns([1,1])
with col_a:
    st.subheader("▶️ Escuchar original")
    st.audio(audio_wav_bytes(audio,fs),format="audio/wav")
with col_b:
    st.subheader("🎚️ Escuchar filtrado")
    st.audio(audio_wav_bytes(filtered,fs),format="audio/wav")

first,second,third,fourth=st.tabs([
    "🔎 Mapa del sonido", "🎚️ Bandas y mezcla", "🎤 Voces e instrumentos IA", "📦 Exportar"
])

with first:
    st.plotly_chart(create_waveform(audio,fs,start),use_container_width=True)
    cp1,cp2,cp3=st.columns([1,1,1])
    with cp1:
        max_display=int(min(12000,fs/2))
        opts=[f for f in [300,1000,4000,8000,12000] if f<=max_display]
        if not opts: opts=[max_display]
        max_freq=st.selectbox("Zoom en frecuencias",opts,index=0,
                              format_func=lambda z:f"0–{z:,} Hz")
    with cp2:
        nfft=st.selectbox("Ventana STFT",[1024,2048,4096,8192],index=2)
    with cp3:
        st.metric("Espaciado entre bins de STFT",f"{min(fs,24000)/nfft:.2f} Hz")
    with st.spinner("Calculando espectrogramas..."):
        original_spec=get_spectrogram(audio,fs,nfft)
        filtered_spec=get_spectrogram(filtered,fs,nfft)
    reference = float(max(np.max(original_spec.magnitude),1e-12))
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(create_spectrogram_chart(original_spec,reference,max_freq,start,"Original"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(create_spectrogram_chart(filtered_spec,reference,max_freq,start,"Después del filtro"),
                        use_container_width=True)
    st.caption("Misma escala en dB para comparar ambos espectrogramas. Puedes acercar regiones con el ratón.")
    st.plotly_chart(create_band_energy(original_spec,start),use_container_width=True)
    with st.expander("📘 Interpretación matemática"):
        st.latex(r"X[m,k]=\sum_n x[n]w[n-mR]e^{-j2\pi kn/N}")
        st.markdown("La STFT utiliza ventanas temporales. El espaciado `Δf=fs/N` no garantiza por sí solo distinguir tonos cercanos. "
                    "La energía por banda identifica rangos frecuenciales, **no demuestra qué instrumento los produce**.")
        st.markdown("Con `sosfiltfilt` la fase neta se cancela, pero la magnitud efectiva del filtro es `|H|²`. ")
    with st.expander("🥁 Ataques y BPM (estimación)"):
        if st.button("Calcular ataques y tempo",key="bpm"):
            with st.spinner("Calculando..."):
                events,bpm=onset_and_tempo(audio,fs)
            st.write(f"**Ataques detectados:** {len(events)}")
            st.write(f"**Tempo estimado:** {bpm:.1f} BPM" if np.isfinite(bpm) else "Tempo no disponible")
            st.caption("Un ataque no equivale necesariamente a una percusión; el BPM puede aparecer a la mitad o el doble.")
            if len(events):
                fig=go.Figure(go.Scatter(x=start+events,y=np.ones_like(events),mode="markers",
                                         marker=dict(color="#17bfa7",size=8)))
                fig.update_layout(title="Instantes con ataques detectados",height=190,
                                  yaxis=dict(visible=False),xaxis_title="Tiempo de canción (s)")
                st.plotly_chart(fig,use_container_width=True)

with second:
    st.subheader("Escucha distintas bandas")
    st.write("Los filtros seleccionan **frecuencias**, no instrumentos: una voz y una guitarra pueden compartir la misma banda.")
    supported={name:lim for name,lim in BANDS.items() if lim[0] < .97*fs/2}
    options=list(supported)
    bands_to_play=st.multiselect("Selecciona las bandas que quieres combinar",options,
                                 default=options[:2])
    if bands_to_play:
        mix=np.zeros_like(audio)
        for name in bands_to_play:
            lo_b,hi_b=supported[name]
            band=filter_band(audio,fs,lo_b,hi_b,order=4)
            if band is not None:
                mix+=band
        st.write("**Mezcla de bandas seleccionadas:**")
        # Ganancia compartida: no normalizamos cada banda por separado.
        mix_gain=min(1.,.98/max(1.,np.max(np.abs(mix))))
        st.audio(audio_wav_bytes(mix,fs,mix_gain),format="audio/wav")
        st.download_button("Descargar mezcla WAV",audio_wav_bytes(mix,fs,mix_gain),
                           file_name="bandas_seleccionadas.wav",mime="audio/wav")
    st.divider()
    for i,(name,(lo_b,hi_b)) in enumerate(supported.items()):
        b=filter_band(audio,fs,lo_b,hi_b,order=4)
        if b is None: continue
        with st.container():
            lc,rc=st.columns([1,3])
            with lc:
                st.write(f"**{name}**")
                st.caption("Contenido frecuencial estimado")
            with rc:
                st.audio(audio_wav_bytes(b,fs),format="audio/wav")
    with st.expander("¿Y las voces?"):
        st.info("La zona vocal aproximada (200–4000 Hz) también contiene guitarra, piano y percusión. "
                "Para aislar voz de verdad utiliza la pestaña 'Voces e instrumentos IA'.")

with third:
    st.subheader("Separación estimada de voces e instrumentos")
    st.write("Para extraer pistas separadas utilizamos **Demucs**. "
             "Esta operación es opcional, consume memoria y necesita descargar los pesos del modelo la primera vez.")
    available = importlib.util.find_spec("demucs") is not None
    if not available:
        st.warning("Demucs no está instalado. En la terminal ejecuta: `pip install demucs` "
                   "(además puede requerir FFmpeg y dependencias de PyTorch).")
    model=st.selectbox("Modelo de separación",["htdemucs_6s","htdemucs"],
                       help="6 fuentes añade guitarra y piano; el piano tiene más artefactos.")
    st.caption("En el modo de 6 fuentes, «cuerdas» significa **guitarra estimada**, no violín/violonchelo.")
    if "stems" not in st.session_state:
        st.session_state["stems"]={}
    cache_key=(str(song_path),round(start,2),round(end,2),model)
    if st.button("Separar el fragmento con IA",disabled=not available, type="primary"):
        with st.spinner("Separando el fragmento. La primera ejecución puede descargar el modelo..."):
            tmp=Path(st.session_state.get("session_audio_dir") or tempfile.mkdtemp(prefix="lab_musical_"))
            st.session_state["session_audio_dir"]=str(tmp)
            wav=tmp/"fragmento_para_ia.wav"
            stereo = audio if audio.shape[0]==2 else np.vstack([audio[0],audio[0]])
            sf.write(str(wav),stereo.T,fs,format="WAV")
            out=tmp/"modelos"
            cmd=[sys.executable,"-m","demucs","-n",model,"--out",str(out),str(wav)]
            try:
                p=subprocess.run(cmd,text=True,capture_output=True,timeout=900,check=True)
                folder=out/model/wav.stem
                stems={}
                for name in ["vocals","drums","bass","other","guitar","piano"]:
                    stem=folder/f"{name}.wav"
                    if stem.is_file():
                        a,stem_fs=sf.read(stem,dtype="float32",always_2d=True)
                        stems[name]=a.mean(axis=1)
                if not {"vocals","drums","bass","other"}.issubset(stems):
                    raise RuntimeError("No se generaron todas las pistas esperadas.")
                st.session_state["stems"][cache_key]=(stems,stem_fs)
                st.success("Separación terminada; escucha los resultados.")
            except Exception as exc:
                st.error(f"La separación falló: {exc}")
                if "p" in locals():
                    with st.expander("Registro técnico"):
                        st.code(p.stderr[-4000:])

    if cache_key in st.session_state["stems"]:
        stems,stem_fs=st.session_state["stems"][cache_key]
        merged,frac=ai_map_data(stems,stem_fs)
        relt=np.arange(frac.shape[1])*.5+start
        fig=go.Figure(go.Heatmap(z=frac,x=relt,y=list(merged.keys()),
                                  colorscale="Viridis",zmin=0,zmax=1,
                                  colorbar=dict(title="Fracción"),
                                  hovertemplate="t=%{x:.1f}s<br>%{y}<br>%{z:.0%}<extra></extra>"))
        fig.update_layout(height=310,title="Mapa de protagonismo estimado (ventanas de 0.5 s)",
                          xaxis_title="Tiempo (s)",margin=dict(l=10,r=10,t=55,b=20))
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("🎚️ Mezclar instrumentos separados")
        active_stems=st.multiselect("Selecciona las fuentes que quieres escuchar juntas",
                                    list(merged),default=list(merged)[:2])
        if active_stems:
            mixed=np.sum([merged[name] for name in active_stems],axis=0)
            gain=min(1.,.98/max(1.,float(np.max(np.abs(mixed)))))
            st.audio(audio_wav_bytes(mixed,stem_fs,gain),format="audio/wav")
            st.download_button("Descargar mezcla de fuentes",audio_wav_bytes(mixed,stem_fs,gain),
                               file_name="mezcla_instrumentos.wav",mime="audio/wav")
        display_source=st.selectbox("Ver espectrograma de una fuente",list(merged))
        s_stem=spectrogram(merged[display_source],stem_fs,n_fft=4096)
        original_for_ref=spectrogram(audio,fs,n_fft=4096)
        st.plotly_chart(create_spectrogram_chart(s_stem,float(np.max(original_for_ref.magnitude)),
                                                int(min(8000,s_stem.sr/2)),start,
                                                f"Espectrograma: {display_source}"),
                        use_container_width=True)
        for label,stem in merged.items():
            st.write(f"**{label}**")
            st.audio(audio_wav_bytes(stem,stem_fs),format="audio/wav")
            st.download_button(f"Descargar {label}",audio_wav_bytes(stem,stem_fs),
                               file_name=f"fuente_{list(merged).index(label)+1}.wav",mime="audio/wav",key="dl"+label)
        st.caption("Proporción de energía no equivale a probabilidad de que el instrumento esté presente. "
                   "La separación puede contener filtraciones y artefactos.")
    else:
        st.info("La separación de fuentes no se ha ejecutado para el fragmento actual. "
                "Puedes usar los espectrogramas y filtros de las otras pestañas sin instalar Demucs.")

with fourth:
    st.subheader("Descargar archivos del análisis")
    st.write("Guarda los dos audios que estás escuchando y la configuración reproducible de los filtros.")
    meta={"audio":source_name,"inicio_s":start,"fin_s":end,"fs_hz":fs,
          "tipo_filtro":kind,"orden":order,"fc_hz":fc,"banda_hz":[lo,hi] if kind=="Pasa-banda" else None,
          "aviso":"La selección por bandas no demuestra identidad instrumental."}
    st.code(json.dumps(meta,ensure_ascii=False,indent=2),language="json")
    st.download_button("📦 Descargar audios y configuración (.zip)",
                       export_bundle(audio,filtered,fs,start,end,meta),
                       file_name="laboratorio_musical_resultados.zip",mime="application/zip",type="primary")
    st.caption("La canción no se descarga de Internet. El usuario debe subir un archivo que tenga derecho a utilizar.")

st.divider()
st.caption("Proyecto educativo • STFT, scipy.signal y modelos opcionales de separación de fuentes • "
           "No inferir instrumentos solamente a partir de una banda frecuencial")
