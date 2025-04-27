import os
import io
import zipfile
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import time

# ----------------------------
# Funciones principales
# ----------------------------

def load_dicom_series(directory):
    """Carga imágenes DICOM tolerante a errores"""
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                dcm = pydicom.dcmread(path, force=True)
                if hasattr(dcm, 'pixel_array'):
                    dicom_files.append((path, dcm))
            except Exception:
                continue

    if not dicom_files:
        return None, None

    dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 0))
    shape_counts = {}
    for _, dcm in dicom_files:
        shape = dcm.pixel_array.shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    best_shape = max(shape_counts, key=shape_counts.get)
    slices = [d[1].pixel_array for d in dicom_files if d[1].pixel_array.shape == best_shape]

    volume = np.stack(slices)

    sample = dicom_files[0][1]
    spacing = list(getattr(sample, 'PixelSpacing', [1,1])) + [getattr(sample, 'SliceThickness', 1)]
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])
    direction = getattr(sample, 'ImageOrientationPatient', [1,0,0,0,1,0])

    direction_matrix = np.array([
        [direction[0], direction[3], 0],
        [direction[1], direction[4], 0],
        [direction[2], direction[5], 1]
    ])

    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction_matrix
    }
    return volume, volume_info

def load_rtstruct(path):
    """Carga contornos RTSTRUCT"""
    struct = pydicom.dcmread(path)
    structures = {}
    if not hasattr(struct, 'ROIContourSequence'):
        return structures
    roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
    for roi in struct.ROIContourSequence:
        color = np.array(roi.ROIDisplayColor) / 255.0 if hasattr(roi, 'ROIDisplayColor') else np.random.rand(3)
        contours = []
        for contour in roi.ContourSequence:
            pts = np.array(contour.ContourData).reshape(-1, 3)
            contours.append({'points': pts, 'z': np.mean(pts[:,2])})
        structures[roi_names[roi.ReferencedROINumber]] = {'color': color, 'contours': contours}
    return structures

def patient_to_voxel(points, volume_info):
    """Convierte puntos de coordenadas paciente a coordenadas de voxel"""
    spacing = np.array(volume_info['spacing'])
    origin = np.array(volume_info['origin'])
    coords = (points - origin) / spacing
    return coords

def draw_slice(volume, slice_idx, plane, structures, volume_info, window, linewidth=2, show_names=True):
    """Dibuja un corte 2D y superpone contornos"""
    fig, ax = plt.subplots(figsize=(8,8))
    plt.axis('off')

    if plane == 'Axial':
        img = volume[slice_idx,:,:]
    elif plane == 'Coronal':
        img = volume[:,slice_idx,:]
    elif plane == 'Sagital':
        img = volume[:,:,slice_idx]
    else:
        raise ValueError("Plano inválido")

    ww, wc = window
    img = np.clip(img, wc - ww/2, wc + ww/2)
    img = (img - img.min()) / (img.max() - img.min())
    ax.imshow(img, cmap='gray', origin='lower')

    if structures:
        for name, struct in structures.items():
            for contour in struct['contours']:
                voxels = patient_to_voxel(contour['points'], volume_info)
                if plane == 'Axial':
                    mask = np.isclose(voxels[:,2], slice_idx, atol=1)
                    pts = voxels[mask][:, [1,0]]
                elif plane == 'Coronal':
                    mask = np.isclose(voxels[:,1], slice_idx, atol=1)
                    pts = voxels[mask][:, [2,0]]
                elif plane == 'Sagital':
                    mask = np.isclose(voxels[:,0], slice_idx, atol=1)
                    pts = voxels[mask][:, [2,1]]

                if len(pts) >= 3:
                    polygon = patches.Polygon(pts, closed=True, fill=False, edgecolor=struct['color'], linewidth=linewidth)
                    ax.add_patch(polygon)
                    if show_names:
                        center = np.mean(pts, axis=0)
                        ax.text(center[0], center[1], name, color=struct['color'], fontsize=8, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    return fig, ax

# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(layout="wide", page_title="Brachyanalysis Pro")

st.title("🔷 Brachyanalysis - Visualizador DICOM Pro")

uploaded_zip = st.file_uploader("Sube tu ZIP de DICOM + RTSTRUCT", type="zip")

# Presets
presets = {
    'Default': (400, 40),
    'Brain': (80, 40),
    'Bone': (2000, 350),
    'Lung': (1500, -600),
    'Abdomen': (350, 50),
    'Chest': (350, 40)
}

if uploaded_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    volume, volume_info = load_dicom_series(temp_dir)
    structures = None

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                try:
                    dcm = pydicom.dcmread(os.path.join(root, file), stop_before_pixels=True)
                    if getattr(dcm, 'Modality', '') == 'RTSTRUCT':
                        structures = load_rtstruct(os.path.join(root, file))
                        break
                except:
                    continue

    if structures:
        st.success(f"✅ Se cargaron {len(structures)} estructuras.")
    else:
        st.warning("⚠️ No se encontraron estructuras RTSTRUCT.")

    if volume is not None:
        mode = st.sidebar.selectbox("Vista:", ["Axial", "Coronal", "Sagital", "3 vistas"])
        selected_preset = st.sidebar.selectbox("Preset ventana:", list(presets.keys()))
        ww, wc = presets[selected_preset]
        linewidth = st.sidebar.slider("Grosor líneas", 1, 8, 2)
        show_names = st.sidebar.checkbox("Mostrar nombres estructuras", value=True)

        if mode != "3 vistas":
            if mode == "Axial":
                n = volume.shape[0]
            elif mode == "Coronal":
                n = volume.shape[1]
            elif mode == "Sagital":
                n = volume.shape[2]

            slice_idx = st.slider(f"Slice {mode}", 0, n-1, n//2)

            playing = st.checkbox("Play/Pause", value=False)
            speed = st.slider("Velocidad (segundos por corte)", 0.05, 1.0, 0.2)

            if playing:
                placeholder = st.empty()
                for idx in range(slice_idx, n):
                    fig, ax = draw_slice(volume, idx, mode, structures, volume_info, (ww, wc), linewidth, show_names)
                    placeholder.pyplot(fig)
                    time.sleep(speed)
                    if not st.session_state.get('playing', True):
                        break
            else:
                fig, ax = draw_slice(volume, slice_idx, mode, structures, volume_info, (ww, wc), linewidth, show_names)
                st.pyplot(fig)

        else:
            slices = [volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2]
            planes = ["Axial", "Coronal", "Sagital"]
            cols = st.columns(3)
            for i, plane in enumerate(planes):
                with cols[i]:
                    fig, ax = draw_slice(volume, slices[i], plane, structures, volume_info, (ww, wc), linewidth, show_names)
                    st.pyplot(fig)
    else:
        st.error("❌ No se encontró serie DICOM válida.")

