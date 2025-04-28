import os
import io
import zipfile
import tempfile
import numpy as np
import streamlit as st
import pydicom
import altair as alt
import pandas as pd

# Configuración de Streamlit
st.set_page_config(page_title="Brachyanalysis", layout="wide")

# --- Funciones auxiliares ---

def extract_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def find_dicom_series(directory):
    series = {}
    structures = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                dcm = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                modality = getattr(dcm, 'Modality', '')
                if modality == 'RTSTRUCT' or file.startswith('RS'):
                    structures.append(path)
                elif modality in ['CT', 'MR', 'PT', 'US']:
                    uid = getattr(dcm, 'SeriesInstanceUID', 'unknown')
                    if uid not in series:
                        series[uid] = []
                    series[uid].append(path)
            except Exception:
                pass
    return series, structures

def load_dicom_series(file_list):
    dicom_files = []
    for file_path in file_list:
        try:
            dcm = pydicom.dcmread(file_path, force=True)
            if hasattr(dcm, 'pixel_array'):
                dicom_files.append((file_path, dcm))
        except Exception:
            continue

    if not dicom_files:
        return None, None

    dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 0))
    slices = [d[1].pixel_array for d in dicom_files]

    volume = np.stack(slices)

    sample = dicom_files[0][1]
    pixel_spacing = list(map(float, getattr(sample, 'PixelSpacing', [1,1])))
    slice_thickness = float(getattr(sample, 'SliceThickness', 1))
    spacing = pixel_spacing + [slice_thickness]
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])

    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'size': volume.shape
    }

    return volume, volume_info

def apply_window(img, center, width):
    img = img.astype(np.float32)
    min_val = center - width / 2
    max_val = center + width / 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    return img

def draw_slice_image(volume, slice_idx, plane, window_center, window_width):
    if plane == 'axial':
        img = volume[slice_idx, :, :]
    elif plane == 'coronal':
        img = volume[:, slice_idx, :]
    elif plane == 'sagittal':
        img = volume[:, :, slice_idx]
    else:
        raise ValueError("Plano inválido")

    img = apply_window(img, window_center, window_width)
    img = np.flipud(img)
    return img

def load_rtstruct_to_dataframe(structure_path, volume_info, plane, slice_idx):
    """Convierte contornos RTSTRUCT a DataFrame de puntos para Altair"""
    try:
        struct = pydicom.dcmread(structure_path)
        df_list = []

        if not hasattr(struct, 'ROIContourSequence'):
            return pd.DataFrame()

        roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
        spacing = np.array(volume_info['spacing'])
        origin = np.array(volume_info['origin'])

        # Coordenada física actual de corte
        if plane == 'axial':
            coord_plane = origin[2] + spacing[2] * slice_idx
            tolerance = spacing[2]
        elif plane == 'coronal':
            coord_plane = origin[1] + spacing[1] * slice_idx
            tolerance = spacing[1]
        elif plane == 'sagittal':
            coord_plane = origin[0] + spacing[0] * slice_idx
            tolerance = spacing[0]

        for roi in struct.ROIContourSequence:
            roi_name = roi_names.get(roi.ReferencedROINumber, f"ROI-{roi.ReferencedROINumber}")
            if not hasattr(roi, 'ContourSequence'):
                continue
            for contour in roi.ContourSequence:
                pts = np.array(contour.ContourData).reshape(-1, 3)
                # Seleccionar contornos cercanos al corte actual
                if plane == 'axial':
                    if np.abs(np.mean(pts[:,2]) - coord_plane) > tolerance:
                        continue
                    x = (pts[:,0] - origin[0]) / spacing[0]
                    y = (pts[:,1] - origin[1]) / spacing[1]
                elif plane == 'coronal':
                    if np.abs(np.mean(pts[:,1]) - coord_plane) > tolerance:
                        continue
                    x = (pts[:,0] - origin[0]) / spacing[0]
                    y = (pts[:,2] - origin[2]) / spacing[2]
                elif plane == 'sagittal':
                    if np.abs(np.mean(pts[:,0]) - coord_plane) > tolerance:
                        continue
                    x = (pts[:,1] - origin[1]) / spacing[1]
                    y = (pts[:,2] - origin[2]) / spacing[2]

                df_contour = pd.DataFrame({'x': x, 'y': y, 'label': roi_name})
                df_list.append(df_contour)

        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            return full_df
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error cargando RTSTRUCT: {e}")
        return pd.DataFrame()

def draw_contours_altair(points_df, width, height):
    if points_df.empty:
        return None
    chart = alt.Chart(points_df).mark_line().encode(
        x='x:Q',
        y='y:Q',
        color='label:N'
    ).properties(
        width=width,
        height=height
    )
    return chart

# --- Interfaz principal Streamlit ---

st.title("🧠 Brachyanalysis Ultra-Rápido con Contornos RTSTRUCT")

uploaded_file = st.sidebar.file_uploader("Sube ZIP con DICOM + RTSTRUCT", type="zip")

if uploaded_file:
    temp_dir = extract_zip(uploaded_file)
    series_dict, structure_files = find_dicom_series(temp_dir)

    if series_dict:
        series_options = list(series_dict.keys())
        selected_series = st.sidebar.selectbox("Selecciona serie", series_options)
        dicom_files = series_dict[selected_series]
        volume, volume_info = load_dicom_series(dicom_files)

        if volume is not None:
            st.sidebar.markdown("## Visualización")
            plane = st.sidebar.radio("Plano", ["axial", "coronal", "sagittal"])

            if plane == 'axial':
                max_idx = volume.shape[0] - 1
            elif plane == 'coronal':
                max_idx = volume.shape[1] - 1
            else:
                max_idx = volume.shape[2] - 1

            slice_idx = st.sidebar.slider(f"Índice {plane}", 0, max_idx, max_idx//2)

            window_center = st.sidebar.number_input("Window Center", value=40)
            window_width = st.sidebar.number_input("Window Width", value=400)

            img = draw_slice_image(volume, slice_idx, plane, window_center, window_width)

            st.image(img, caption=f"{plane.capitalize()} slice {slice_idx}", use_column_width=True, clamp=True)

            if structure_files:
                points_df = load_rtstruct_to_dataframe(structure_files[0], volume_info, plane, slice_idx)
                chart = draw_contours_altair(points_df, width=600, height=600)
                if chart:
                    st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("No se encontraron imágenes DICOM.")
