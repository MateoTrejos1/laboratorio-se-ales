# Laboratorio musical interactivo | Señales y Sistemas

Aplicación educativa para visualizar y escuchar diferentes bandas de frecuencia en archivos de audio propios o autorizados. Implementa análisis espectrotemporal mediante STFT, filtrado digital Butterworth (SciPy), visualización interactiva de espectrogramas, análisis de energía y reproducción de los fragmentos originales y filtrados. Si no se adjunta audio, ofrece una demostración **sintética**, claramente etiquetada.

## Abrir la aplicación

Cuando la aplicación esté publicada, añade aquí su URL de Streamlit Community Cloud para que el docente pueda abrirla.

## Instrucciones de uso

1. Abre la aplicación web.
2. Prueba la **demo sintética** integrada o sube un archivo MP3, WAV o FLAC que tengas derecho a utilizar.
3. En la barra lateral selecciona un fragmento breve, idealmente entre 10 y 25 segundos.
4. Cambia el filtro a pasa-bajas, elige una frecuencia de corte (p. ej. 120 Hz) y compara el audio original con el filtrado.
5. Explora **Mapa del sonido** para visualizar los espectrogramas, y **Bandas y mezcla** para escuchar subgraves, graves, medios y agudos.
6. En **Exportar**, descarga los fragmentos y parámetros del análisis.

## Fundamento técnico

- STFT: `X[m,k] = sum_n x[n] w[n-mR] exp(-j 2pi kn/N)`.
- Resolución de la rejilla FFT: `Delta f = fs/N` (distinta de la capacidad real de separar componentes cercanas).
- Filtro IIR Butterworth de orden configurable, con `scipy.signal.sosfiltfilt` para fase neta cero en análisis fuera de línea (puede presentar efectos de borde). La magnitud efectiva es `|H|^2`.
- La separación por **bandas de frecuencia** NO equivale a la identificación de voces o instrumentos.

## Instalación local (opcional)

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publicación desde GitHub

1. Sube los archivos a un repositorio público de GitHub.
2. Abre <https://share.streamlit.io>, conéctalo a GitHub y elige **Create app**.
3. Selecciona tu repositorio, rama `main` y archivo principal `app.py`.
4. Selecciona Python 3.12 en ajustes avanzados si necesitas especificar versión.
5. Publica y copia el enlace `*.streamlit.app`.

## Alcance y límites

- La versión básica publicada **no incluye la ejecución del modelo Demucs**. La pestaña de separación IA muestra cómo habilitarlo; requiere dependencias y recursos computacionales adicionales que pueden superar los límites gratuitos del alojamiento.
- El código fuente no incluye grabaciones comerciales, ni debe subirse al repositorio música protegida sin permiso. El audio que sube un usuario para analizar se procesa temporalmente en el servidor del alojamiento.
- La exactitud de detección de percusión/BPM y de la separación por IA es aproximada, no una clasificación infalible.
