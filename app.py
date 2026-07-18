import os
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import time
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image as PILImage
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import subprocess
import sys
import cv2

TRADUCCION_CLASES = {
    "safe-pothole": "Bajo Riesgo",
    "medium-pothole": "Riesgo Medio",
    "risk-pothole": "Alto Riesgo"
}

st.set_page_config(
    page_title="Detector de Baches",
    layout="wide",
    initial_sidebar_state="expanded"
)


API_KEY = "ar4ePdc7mmldDifuoth4"
MODEL_ID = "pothole-class-2/2"

COLORES = {
    "safe-pothole":   (255, 0, 0),
    "medium-pothole": (128, 0, 128),
    "risk-pothole":   (0, 200, 0),
}

COLORES_MPL = {
    "safe-pothole":   "#FF0000",
    "medium-pothole": "#800080",
    "risk-pothole":   "#00BB00",
}

tab_detector, tab_ayuda, tab_reconocimientos = st.tabs(
    ["Detector de Baches", "Ayuda", "Reconocimientos"]
)

def calcular_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    interseccion = max(0, x2 - x1) * max(0, y2 - y1)
    if interseccion == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - interseccion
    return interseccion / union if union > 0 else 0.0

def es_duplicado(nueva_box, baches_previos, threshold):
    for box_previa in baches_previos:
        if calcular_iou(nueva_box, box_previa) > threshold:
            return True
    return False

def buf_a_archivo(buf):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(buf.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def generar_pie_chart(contadores):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")
    valores = list(contadores.values())
    etiquetas = [TRADUCCION_CLASES[k] for k in contadores.keys()]
    colores = [COLORES_MPL[k] for k in contadores.keys()]
    wedges, texts, autotexts = ax.pie(
        valores, labels=etiquetas, colors=colores,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "black", "fontsize": 10}
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(9)
    ax.set_title("Distribución de baches por tipo", color="black", fontsize=12, pad=15)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf

def generar_barras(contadores):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("#f5f5f5")
    clases = list(contadores.keys())
    valores = list(contadores.values())
    colores = [COLORES_MPL[k] for k in clases]
    clases_es = [TRADUCCION_CLASES[k] for k in clases]
    bars = ax.barh(clases_es, valores, color=colores, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va="center", color="black", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cantidad", color="black")
    ax.set_title("Baches detectados por tipo", color="black", fontsize=12)
    ax.tick_params(colors="black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(valores) * 1.2 if max(valores) > 0 else 1)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf

def generar_linea_tiempo(detecciones_por_segundo, fps):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.set_facecolor("#f5f5f5")
    segundos = sorted(detecciones_por_segundo.keys())
    for clase, color in COLORES_MPL.items():
        valores = [detecciones_por_segundo.get(s, {}).get(clase, 0) for s in segundos]
        ax.plot(segundos, valores, color=color, linewidth=2, label=TRADUCCION_CLASES[clase])
        ax.fill_between(segundos, valores, alpha=0.15, color=color)
    ax.set_xlabel("Tiempo (segundos)", color="black")
    ax.set_ylabel("Detecciones", color="black")
    ax.set_title("Detecciones a lo largo del video", color="black", fontsize=12)
    ax.tick_params(colors="black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="white", labelcolor="black")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf

def generar_mapa_calor(todas_las_boxes, ancho, alto):
    mapa = np.zeros((alto, ancho), dtype=np.float32)
    for box, clase in todas_las_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ancho, x2), min(alto, y2)
        if x2 > x1 and y2 > y1:
            peso = 3 if clase == "risk-pothole" else 2 if clase == "medium-pothole" else 1
            mapa[y1:y2, x1:x2] += peso
    if mapa.max() > 0:
        mapa = mapa / mapa.max()
    mapa_blur = cv2.GaussianBlur(mapa, (51, 51), 0)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    im = ax.imshow(mapa_blur, cmap="hot", aspect="auto", interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, label="Intensidad")
    cbar.ax.yaxis.label.set_color("black")
    cbar.ax.tick_params(colors="black")
    ax.set_title("Mapa de calor de baches (risk=3x, medium=2x, safe=1x)",
                 color="black", fontsize=11)
    ax.tick_params(colors="black")
    ax.set_xlabel("Ancho del frame (px)", color="black")
    ax.set_ylabel("Alto del frame (px)", color="black")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf

def generar_indicador_severidad(contadores):
    total = sum(contadores.values())
    score = 0
    if total > 0:
        score = (
            contadores["risk-pothole"] * 3 +
            contadores["medium-pothole"] * 2 +
            contadores["safe-pothole"] * 1
        ) / (total * 3) * 100

    if score < 33:
        nivel = "NORMAL"
        color_nivel = "#00AA00"
    elif score < 66:
        nivel = "MODERADO"
        color_nivel = "#FF8800"
    else:
        nivel = "CRÍTICO"
        color_nivel = "#CC0000"

    fig, ax = plt.subplots(figsize=(8, 2), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.barh(0.5, 100, height=0.4, color="#dddddd", left=0)
    ax.barh(0.5, min(score, 33), height=0.4, color="#00AA00", left=0)
    if score > 33:
        ax.barh(0.5, min(score - 33, 33), height=0.4, color="#FF8800", left=33)
    if score > 66:
        ax.barh(0.5, score - 66, height=0.4, color="#CC0000", left=66)
    ax.axvline(x=score, color="black", linewidth=2, linestyle="--")
    ax.text(score, 0.95, f"{score:.1f}%", ha="center", va="top",
            color="black", fontsize=11, fontweight="bold")
    ax.text(50, 0.05, f"Nivel de severidad: {nivel}", ha="center", va="bottom",
            color=color_nivel, fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.set_title("Indicador de severidad de la carretera", color="black", fontsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf

def generar_pdf(contadores, duracion, total_frames, fps,
                detecciones_por_segundo, todas_las_boxes,
                hallazgos_risk, ancho, alto):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)
    pdf.cell(0, 15, "Reporte de Estado de Carretera", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.cell(0, 8, f"Duración del video: {duracion:.1f} segundos", ln=True, align="C")
    pdf.cell(0, 8, f"Frames analizados: {total_frames}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Resúmen de detecciones:", ln=True)
    pdf.set_font("Arial", "", 12)
    total = sum(contadores.values())
    colores_texto = {
        "safe-pothole":   (200, 0, 0),
        "medium-pothole": (120, 0, 120),
        "risk-pothole":   (0, 150, 0)
    }
    for clase, cantidad in contadores.items():
        pct = (cantidad / total * 100) if total > 0 else 0
        r, g, b = colores_texto[clase]
        pdf.set_text_color(r, g, b)
        nombre_traducido = TRADUCCION_CLASES[clase]
        pdf.cell(0, 10, f"  {nombre_traducido}: {cantidad} ({pct:.1f}%)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"  Total: {total} baches unicos detectados", ln=True)
    pdf.ln(5)

    nombre_sev = buf_a_archivo(generar_indicador_severidad(contadores))
    pdf.image(nombre_sev, x=10, w=190)
    os.remove(nombre_sev)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.cell(0, 12, "Gráficas de Distribución", ln=True, align="C")
    pdf.ln(5)
    nombre_pie = buf_a_archivo(generar_pie_chart(contadores))
    pdf.image(nombre_pie, x=30, w=150)
    os.remove(nombre_pie)
    pdf.ln(5)
    nombre_barras = buf_a_archivo(generar_barras(contadores))
    pdf.image(nombre_barras, x=30, w=150)
    os.remove(nombre_barras)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.cell(0, 12, "Análisis Temporal y Espacial", ln=True, align="C")
    pdf.ln(5)
    nombre_tiempo = buf_a_archivo(generar_linea_tiempo(detecciones_por_segundo, fps))
    pdf.image(nombre_tiempo, x=10, w=190)
    os.remove(nombre_tiempo)
    pdf.ln(5)
    nombre_calor = buf_a_archivo(generar_mapa_calor(todas_las_boxes, ancho, alto))
    pdf.image(nombre_calor, x=10, w=190)
    os.remove(nombre_calor)

    if hallazgos_risk:
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        pdf.cell(0, 12, "Galería de Baches Criticos (Risk)", ln=True, align="C")
        pdf.ln(5)
        col_x = [10, 75, 140]
        fila_y = pdf.get_y()
        col_idx = 0
        for i, hallazgo in enumerate(hallazgos_risk[:12]):
            img_buf = io.BytesIO(hallazgo["imagen_bytes"])
            nombre_crop = buf_a_archivo(img_buf)
            pdf.image(nombre_crop, x=col_x[col_idx], y=fila_y, w=60)
            os.remove(nombre_crop)
            pdf.set_xy(col_x[col_idx], fila_y + 42)
            pdf.set_font("Arial", "", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(60, 5, f"Frame {hallazgo['frame']} | {hallazgo['confianza']:.0%}",
                     ln=False, align="C")
            col_idx += 1
            if col_idx >= 3:
                col_idx = 0
                fila_y += 52
                if fila_y > 240:
                    pdf.add_page()
                    fila_y = 20

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.cell(0, 12, "Recomendaciones", ln=True, align="C")
    pdf.ln(10)
    total = sum(contadores.values())
    score = 0
    if total > 0:
        score = (
            contadores["risk-pothole"] * 3 +
            contadores["medium-pothole"] * 2 +
            contadores["safe-pothole"] * 1
        ) / (total * 3) * 100
    if score >= 66:
        pdf.set_text_color(200, 0, 0)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "NIVEL CRíTICO - INTERVENCIÓN URGENTE", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(60, 60, 60)
        pdf.ln(5)
        pdf.multi_cell(0, 8, "Se detectaron múltiples baches de alto riesgo que representan un peligro inmediato para los usuarios de la vía, se recomienda cerrar parcialmente la vía y realizar reparaciones de emergencia en un plazo máximo de 48 horas.")
    elif score >= 33:
        pdf.set_text_color(200, 100, 0)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "NIVEL MODERADO - MANTENIMIENTO PREVENTIVO", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(60, 60, 60)
        pdf.ln(5)
        pdf.multi_cell(0, 8, "Se detectaron baches de riesgo medio que requieren atención en el corto plazo, se recomienda programar trabajos de mantenimiento en un plazo de 2 semanas para prevenir el deterioro de la vía.")
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "NIVEL NORMAL - ESTADO ACEPTABLE", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(60, 60, 60)
        pdf.ln(5)
        pdf.multi_cell(0, 8, "El estado general de la carretera es aceptable, se recomienda realizar inspecciones periodicas cada 3 meses para monitorear el estado de la via y prevenir deterioro futuro.")

    return bytes(pdf.output())


def mostrar_resultados():
    contadores = st.session_state["contadores"]
    hallazgos_risk = st.session_state["hallazgos_risk"]
    video_path_final = st.session_state["video_path_final"]
    pdf_bytes = st.session_state["pdf_bytes"]

    st.markdown("---")
    st.markdown("## Resultados Finales")

    _, col1, col2, col3, col4, _ = st.columns([0.5, 1, 1, 1, 1, 0.5])
    total_baches = sum(contadores.values())
    col1.metric("Total baches", total_baches)
    col2.metric("Bajo Riesgo", contadores["safe-pothole"])
    col3.metric("Riesgo Medio", contadores["medium-pothole"])
    col4.metric("Alto Riesgo", contadores["risk-pothole"])

    st.markdown("### Distribución de baches")
    df_contadores = pd.DataFrame({
        "Tipo": [TRADUCCION_CLASES[k] for k in contadores.keys()],
        "Cantidad": list(contadores.values())
    })
    st.bar_chart(df_contadores.set_index("Tipo"))

    st.markdown("### Video procesado")
    col_iz, col_centro, col_der = st.columns([1, 3, 1])
    with col_centro:
        if os.path.exists(video_path_final):
            st.video(video_path_final)
            with open(video_path_final, "rb") as f:
                st.download_button(
                    "Descargar video anotado",
                    f,
                    file_name="video_resultado.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key="descarga_video"
                )
        else:
            st.warning("El archivo de video temporal ya no está disponible. Vuelve a procesar el video.")

    st.markdown("### Galeria de baches críticos (Risk)")
    if hallazgos_risk:
        items_por_fila = 4
        for fila_idx in range(0, len(hallazgos_risk), items_por_fila):
            fila = hallazgos_risk[fila_idx:fila_idx + items_por_fila]
            cols = st.columns(items_por_fila)
            for col_idx, hallazgo in enumerate(fila):
                with cols[col_idx]:
                    img_b64 = base64.b64encode(hallazgo["imagen_bytes"]).decode()
                    st.markdown(
                        f'''
                        <div style="text-align:center;">
                            <img src="data:image/png;base64,{img_b64}"
                                style="width:100%; border-radius:6px;"/>
                            <p style="font-size:11px; color:gray; margin-top:4px;">
                                Frame {hallazgo["frame"]} | {hallazgo["confianza"]:.0%}
                            </p>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            for col_idx in range(len(fila), items_por_fila):
                with cols[col_idx]:
                    st.empty()
    else:
        st.info("No se detectaron baches de riesgo.")

    st.markdown("### Reporte PDF")
    col_iz, col_centro, col_der = st.columns([1, 2, 1])
    with col_centro:
        st.download_button(
            "Descargar Reporte PDF",
            pdf_bytes,
            file_name="reporte_baches.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="descarga_pdf"
        )


with tab_detector:

    st.title("Detector de Baches")
    st.markdown("Sube un video grabado para analizar el estado de la carretera.")
    st.markdown("---")

    col_conf, col_iou = st.columns(2)
    with col_conf:
        confianza = st.slider("Umbral de confianza", 0.1, 1.0, 0.4, 0.01,
                              help="Que tan seguro debe estar el modelo para reportar un bache.")
    with col_iou:
        iou_threshold = st.slider("Umbral IoU (evitar duplicados)", 0.1, 0.9, 0.4, 0.01,
                                  help="Valores más bajos reducen baches duplicados.")

    st.markdown("---")
    video_file = st.file_uploader("Sube tu video", type=["mp4", "avi", "mov"])

    if video_file is not None:

        if st.session_state.get("ultimo_video_nombre") != video_file.name:
            st.session_state["procesado"] = False
            st.session_state["ultimo_video_nombre"] = video_file.name

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duracion = total_frames / fps
        cap.release()

        col1, col2, col3 = st.columns(3)
        col1.metric("Duración", f"{duracion:.1f}s")
        col2.metric("Total frames", total_frames)
        col3.metric("FPS", f"{fps:.1f}")

        procesar = st.button("Procesar Video", type="primary")

        if procesar:

            contadores = {"safe-pothole": 0, "medium-pothole": 0, "risk-pothole": 0}
            hallazgos_risk = []
            baches_unicos = {"safe-pothole": [], "medium-pothole": [], "risk-pothole": []}
            todas_las_boxes = []
            detecciones_por_segundo = {}

            output_path = tempfile.mktemp(suffix="_resultado.mp4")
            out_writer = [None]

            st.markdown("---")
            st.markdown("### Procesando...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            col_m1, col_m2, col_m3 = st.columns(3)
            metric_safe = col_m1.empty()
            metric_medium = col_m2.empty()
            metric_risk = col_m3.empty()

            def render_predictions(predictions, video_frame):
                frame = video_frame.image.copy()
                h, w = frame.shape[:2]

                if out_writer[0] is None:
                    out_writer[0] = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (w, h)
                    )

                progress = min(video_frame.frame_id / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Frame {video_frame.frame_id}/{total_frames}")

                segundo_actual = int(video_frame.frame_id / fps)
                if segundo_actual not in detecciones_por_segundo:
                    detecciones_por_segundo[segundo_actual] = {
                        "safe-pothole": 0, "medium-pothole": 0, "risk-pothole": 0
                    }

                preds_frame = []
                if predictions and "predictions" in predictions:
                    preds_frame = predictions["predictions"]

                boxes_frame = []
                preds_filtradas = []
                for pred in preds_frame:
                    x_c = pred.get("x", 0)
                    y_c = pred.get("y", 0)
                    w_b = pred.get("width", 0)
                    h_b = pred.get("height", 0)
                    nueva_box = (x_c - w_b/2, y_c - h_b/2, x_c + w_b/2, y_c + h_b/2)
                    duplicado_en_frame = False
                    for box_existente, clase_existente in boxes_frame:
                        if (pred.get("class") == clase_existente and
                                calcular_iou(nueva_box, box_existente) > 0.3):
                            duplicado_en_frame = True
                            break
                    if not duplicado_en_frame:
                        boxes_frame.append((nueva_box, pred.get("class")))
                        preds_filtradas.append((pred, nueva_box))

                for pred, nueva_box in preds_filtradas:
                    clase = pred.get("class", "")
                    conf = pred.get("confidence", 0)
                    color = COLORES.get(clase, (255, 255, 0))

                    todas_las_boxes.append((nueva_box, clase))

                    if clase in detecciones_por_segundo[segundo_actual]:
                        detecciones_por_segundo[segundo_actual][clase] += 1

                    if clase in baches_unicos:
                        if not es_duplicado(nueva_box, baches_unicos[clase], iou_threshold):
                            baches_unicos[clase].append(nueva_box)
                            contadores[clase] += 1
                            if clase == "risk-pothole" and len(hallazgos_risk) < 8:
                              x1 = max(0, int(nueva_box[0]))
                              y1 = max(0, int(nueva_box[1]))
                              x2 = min(ancho, int(nueva_box[2]))
                              y2 = min(alto, int(nueva_box[3]))
                              crop = frame[y1:y2, x1:x2]
                              if crop.size > 0:
                                  crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                  img_pil = PILImage.fromarray(crop_rgb)
                                  img_pil = img_pil.resize((300, 200), PILImage.LANCZOS)
                                  buf = io.BytesIO()
                                  img_pil.save(buf, format="PNG")
                                  buf.seek(0)
                                  hallazgos_risk.append({
                                      "imagen_bytes": buf.getvalue(),
                                      "frame": video_frame.frame_id,
                                      "confianza": conf
                                  })

                    if "points" in pred:
                        puntos = np.array(
                            [[int(p["x"]), int(p["y"])] for p in pred["points"]],
                            np.int32
                        )
                        cv2.polylines(frame, [puntos], isClosed=True, color=color, thickness=3)
                        x, y = puntos[0]
                        clase_es = TRADUCCION_CLASES.get(clase, clase)
                        label = f"{clase_es} {conf:.0%}"
                        x = max(0, min(x, w - len(label) * 13))
                        y = max(30, y)
                        cv2.rectangle(frame, (x, y-30), (x + len(label)*13, y), color, -1)
                        cv2.putText(frame, label, (x, y-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                out_writer[0].write(frame)
                metric_safe.metric("Safe", contadores["safe-pothole"])
                metric_medium.metric("Medium", contadores["medium-pothole"])
                metric_risk.metric("Risk", contadores["risk-pothole"])

            try:
                pipeline = InferencePipeline.init(
                    model_id=MODEL_ID,
                    video_reference=video_path,
                    on_prediction=render_predictions,
                    api_key=API_KEY,
                    confidence=confianza
                )
                pipeline.start()
                pipeline.join()
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if out_writer[0]:
                    out_writer[0].release()

            progress_bar.progress(1.0)
            status_text.text("Procesamiento completado")

            with st.spinner("Convirtiendo video..."):
                try:
                    import imageio
                    output_h264 = output_path.replace(".mp4", "_h264.mp4")
                    reader = imageio.get_reader(output_path)
                    fps_vid = reader.get_meta_data()["fps"]
                    writer = imageio.get_writer(output_h264, fps=fps_vid, codec="libx264")
                    for frame in reader:
                        writer.append_data(frame)
                    writer.close()
                    reader.close()

                    video_path_final = output_h264
                except Exception as e:
                    st.warning(f"No se pudo convertir: {e}")
                    video_path_final = output_path

            with st.spinner("Generando PDF..."):
                pdf_bytes = generar_pdf(
                    contadores=contadores,
                    duracion=duracion,
                    total_frames=total_frames,
                    fps=fps,
                    detecciones_por_segundo=detecciones_por_segundo,
                    todas_las_boxes=todas_las_boxes,
                    hallazgos_risk=hallazgos_risk,
                    ancho=ancho,
                    alto=alto
                )

            st.session_state["contadores"] = contadores
            st.session_state["hallazgos_risk"] = hallazgos_risk
            st.session_state["output_path"] = output_path
            st.session_state["video_path_final"] = video_path_final
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["duracion"] = duracion
            st.session_state["total_frames"] = total_frames
            st.session_state["fps"] = fps
            st.session_state["detecciones_por_segundo"] = detecciones_por_segundo
            st.session_state["todas_las_boxes"] = todas_las_boxes
            st.session_state["ancho"] = ancho
            st.session_state["alto"] = alto
            st.session_state["procesado"] = True

    if st.session_state.get("procesado", False):
        mostrar_resultados()

with tab_ayuda:
    st.title("Ayuda")
    st.markdown("---")

    st.markdown("## ¿Cómo funciona el sistema?")
    st.markdown("""
    Este sistema utiliza inteligencia artificial para detectar y clasificar baches
    en videos grabados desde dron o cámara. El modelo analiza cada frame del video
    e identifica tres tipos de baches según su nivel de riesgo.
    """)

    st.markdown("---")
    st.markdown("## Clasificación de baches")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Safe")
        st.markdown("""
        **Color:** Rojo

        Baches de bajo riesgo: Son pequeños o superficiales y no representan
        un peligro inmediato para los usuarios de la vía.
        """)
    with col2:
        st.markdown("### Medium")
        st.markdown("""
        **Color:** Morado

        Baches de riesgo moderado: Requieren atención en el corto plazo
        para evitar que se conviertan en baches de alto riesgo.
        """)
    with col3:
        st.markdown("### Risk")
        st.markdown("""
        **Color:** Verde

        Baches de alto riesgo: Baches grandes o profundos y representan
        un peligro real para vehículos y personas. Requieren atención urgente.
        """)

    st.markdown("---")
    st.markdown("## Guía de uso")

    st.markdown("### Paso 1 — Configurar parámetros")
    st.markdown("""
    Antes de procesar el video ajusta los dos controles disponibles:

    - **Umbral de confianza:** Que tan seguro debe estar el modelo para reportar un bache.
      Valores altos (0.7-0.9) muestran solo detecciones muy seguras.
      Valores bajos (0.2-0.4) muestran más detecciones pero pueden incluir falsos positivos.

    - **Umbral IoU:** Controla cuántos baches duplicados se filtran.
      Valores bajos (0.1-0.3) son más estrictos y eliminan mas duplicados.
      Valores altos (0.6-0.9) permiten contar más baches aunque sean similares.
    """)

    st.markdown("### Paso 2 — Subir el video")
    st.markdown("""
    Haz clic en el botón de carga y selecciona tu video.
    Se aceptan formatos MP4, AVI y MOV.
    Se mostraran datos básicos del video como duración, total de frames y FPS.
    """)

    st.markdown("### Paso 3 — Procesar")
    st.markdown("""
    Haz clic en el boton **Procesar Video**. Durante el procesamiento verás:

    - Barra de progreso con el frame actual
    - Contadores en tiempo real de cada tipo de bache
    """)

    st.markdown("### Paso 4 — Revisar resultados")
    st.markdown("""
    Al terminar el procesamiento se mostraran:

    - **Métricas finales:** Total de baches únicos por tipo
    - **Gráfica de distribución:** Comparación visual de los tipos de baches
    - **Video procesado:** Puedes ver el video con las anotaciones dibujadas o descargarlo
    - **Galería de baches críticos:** Fotos recortadas de los baches de tipo Risk
    - **Reporte PDF:** Documento descargable con gráficas, mapa de calor y recomendaciones
    """)

    st.markdown("---")
    st.markdown("## Preguntas frecuentes")

    with st.expander("El modelo detecto demasiados baches duplicados, ¿qué hago?"):
        st.markdown("Baja el umbral IoU, un valor de 0.2 o 0.3 filtra la mayoria de duplicados.")

    with st.expander("El modelo no detecto ningun bache aunque si los hay, ¿qué hago?"):
        st.markdown("Baja el umbral de confianza, prueba con valores de 0.2 o 0.3 para ver más detecciones.")

    with st.expander("El video procesado no se puede reproducir en la app, ¿qué hago?"):
        st.markdown("Descarga el video y reprodúcelo localmente con cualquier reproductor de video.")

    with st.expander("¿Cuántos créditos consume el sistema?"):
        st.markdown("El sistema usa inferencia local de Roboflow que consume 3,000 frames por crédito. Para un video de 2 minutos a 60fps (aprox. 7,000 frames) se consumen alrededor de 2.5 créditos.")

    with st.expander("¿El sistema funciona con videos de baja calidad?"):
        st.markdown("Sí, pero la precisión mejora notablemente con videos de mayor resolución y grabados con luz natural, los videos a 1080p o 4K dan mejores resultados.")

with tab_reconocimientos:
    st.title("Reconocimientos")
    st.markdown("---")

    st.markdown("## Institucional")
    st.markdown("""
    Este proyecto fue desarrollado como parte de un proyecto tecnológico en la
    **Universidad Autónoma Metropolitana Unidad Azcapotzalco**, agradezco profundamente el apoyo
    institucional brindado para la realización de este sistema.
    """)

    st.markdown("---")
    st.markdown("## Asesoras")
    st.markdown("""
    Expreso mi más sincero agradecimiento a mis asesoras por su
    orientacion, dedicacion y apoyo a lo largo del desarrollo de este proyecto:

    - **Dra. Silvia Beatriz González Brambila** — Profesora Titular
    - **Dra. Beatriz Adriana González Beltrán** — Profesora Titular
    """)

    st.markdown("---")
    st.markdown("## Equipo de desarrollo")
    st.markdown("""
    Este sistema fue desarrollado por:

    - **Edgar Angeles Rodríguez**
    """)

    st.markdown("---")
    st.markdown("## Tecnologías utilizadas")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Modelo**")
        st.markdown("""
        - Roboflow
        - YOLO26
        - Ultralytics
        """)
    with col2:
        st.markdown("**Desarrollo**")
        st.markdown("""
        - Python
        - Streamlit
        - OpenCV
        """)
    with col3:
        st.markdown("**Herramientas**")
        st.markdown("""
        - DJI Phantom 4 Advanced
        - Matplotlib
        """)
