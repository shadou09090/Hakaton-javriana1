"""Interfaz local SpaceLAB. Ejecutar con python -m streamlit run app_streamlit.py."""
import json
from pathlib import Path
import subprocess
import time

import pandas as pd
from PIL import Image, ImageDraw
import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parent
NAMES = ["small_boat", "cargo_ship", "fishing_vessel", "warship", "work_vessel"]
COLORS = ["#ffc107", "#00bcd4", "#ff80ab", "#ef5350", "#76ff03"]

st.set_page_config(page_title="SpaceLAB · Entrenamiento", page_icon="🛰️", layout="wide")
st.title("SpaceLAB · Datos y entrenamiento")
st.caption("YOLO · cinco clases de embarcaciones · equipo A")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data
def load_manifest(path, modified):
    return pd.read_json(path)


@st.fragment(run_every="5s")
def monitor():
    left, right = st.columns(2)
    left.metric("CPU", f"{psutil.cpu_percent(interval=0.1):.0f} %")
    ram = psutil.virtual_memory()
    right.metric("RAM disponible", f"{ram.available/1024**3:.1f} GB")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            st.caption(f"{parts[0]} · GPU {parts[1].strip()} % · VRAM {parts[2].strip()}/{parts[3].strip()} MiB · {parts[4].strip()} °C")
    except (OSError, subprocess.TimeoutExpired):
        st.caption("Telemetría NVIDIA no disponible")
    runs = sorted((ROOT/"runs").glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        st.info("Esperando la primera corrida.")
    for path in runs:
        status = read_json(path)
        if not status:
            continue
        with st.container(border=True):
            st.subheader(status["name"])
            phase = status["phase"]
            if phase in {"training", "evaluating", "initializing"} and not psutil.pid_exists(status["pid"]):
                phase = "interrumpida · proceso ausente"
            st.write(f"Estado · {phase}")
            epochs = status.get("epochs", 1)
            st.progress(min(1.0, status.get("epoch", 0)/max(1, epochs)),
                        text=f"Época {status.get('epoch', 0)} de {epochs}")
            st.caption(f"Última actualización hace {max(0, time.time()-status.get('updated', time.time())):.0f} s")
            if status.get("error"):
                st.error(status["error"])
            metrics = read_json(path.parent/"metricas_val.json")
            if metrics:
                cols = st.columns(3)
                for col, key, title in zip(cols, ["score", "map50_95", "f1_macro"], ["Score oficial", "mAP50–95", "F1 macro"]):
                    col.metric(title, f"{metrics['total'][key]:.4f}")
                st.dataframe(pd.DataFrame(metrics["total"]["por_clase"]).T)
            csv = path.parent/"results.csv"
            if csv.exists():
                try:
                    frame = pd.read_csv(csv)
                    frame.columns = frame.columns.str.strip()
                    columns = [c for c in frame if c.startswith("train/") and "loss" in c]
                    if len(frame) and columns:
                        st.line_chart(frame.set_index("epoch")[columns])
                except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                    st.caption("Se está escribiendo el registro.")


with st.sidebar:
    page = st.radio("Vista", ["Entrenamiento", "Dataset", "Recortes"])

if page == "Entrenamiento":
    monitor()
else:
    development = ROOT/"derived/development"
    manifest_path = development/"manifest.json"
    if not manifest_path.exists():
        st.info("La preparación del dataset todavía está en proceso.")
        st.stop()
    frame = load_manifest(str(manifest_path), manifest_path.stat().st_mtime)
    if page == "Dataset":
        split = st.selectbox("Partición", ["train_dev", "val_dev"])
        selected = frame[frame["split"] == split]
        counts = pd.DataFrame(selected["counts"].tolist(), columns=NAMES).sum()
        st.write(f"{len(selected):,} imágenes · {int(counts.sum()):,} objetos")
        st.bar_chart(counts)
        source = st.selectbox("Fuente", ["Todas"]+sorted(selected["source"].unique()))
        if source != "Todas":
            selected = selected[selected.source == source]
        class_name = st.selectbox("Clase", ["Todas"]+NAMES)
        if class_name != "Todas":
            selected = selected[selected["counts"].map(lambda c: c[NAMES.index(class_name)] > 0)]
        if selected.empty:
            st.info("No hay imágenes para estos filtros.")
            st.stop()
        item = st.selectbox("Imagen", selected["image"].tolist())
        name = Path(item).name
        image_path = development/"images"/split/name
        label_path = development/"labels"/split/(Path(name).stem+".txt")
    else:
        folders = sorted(p for p in (ROOT/"derived").glob("tiles*") if (p/"manifest.json").exists())
        if not folders:
            st.info("Todavía no hay recortes terminados.")
            st.stop()
        folder = st.selectbox("Conjunto de recortes", folders, format_func=lambda p: p.name)
        st.json(read_json(folder/"summary.json"))
        items = sorted((folder/"images/train").glob("*.jpg"))
        image_path = st.selectbox("Recorte", items, format_func=lambda p: p.name)
        label_path = folder/"labels/train"/(image_path.stem+".txt")
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    labels = []
    for row in label_path.read_text().splitlines():
        c, xc, yc, w, h = map(float, row.split())
        c = int(c)
        box = [(xc-w/2)*image.width, (yc-h/2)*image.height,
               (xc+w/2)*image.width, (yc+h/2)*image.height]
        draw.rectangle(box, outline=COLORS[c], width=2)
        draw.text((box[0], max(0, box[1]-12)), NAMES[c], fill=COLORS[c])
        labels.append(dict(clase=NAMES[c], xc=xc, yc=yc, ancho=w, alto=h))
    st.image(image, caption=f"{image_path.name} · {image.width} × {image.height}", width="stretch")
    st.dataframe(pd.DataFrame(labels), hide_index=True)
