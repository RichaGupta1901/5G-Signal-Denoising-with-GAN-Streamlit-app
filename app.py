import streamlit as st
import torch
import numpy as np
import h5py
from model import Generator
from io import BytesIO
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the Generator model
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator_denoising_5g.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Sidebar - About Section
with st.sidebar:
    st.markdown("## About the Project")
    st.markdown("""
    This app demonstrates **5G signal denoising using GANs (Generative Adversarial Networks)**.  
    It accepts noisy I/Q signals in `.h5` or `.npy` formats and applies a trained generator model to produce a denoised version of the signal.

    **Technologies Used:**
    - PyTorch for model inference  
    - Streamlit for UI  
    - Plotly for interactive signal plots
                  
    """)

# UI
st.markdown('<div class="techno-head">5G Signal Denoising with GAN</div>', unsafe_allow_html=True)

# Center the uploader using a container
with st.container():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your noisy signal (.h5 or .npy)", type=["h5", "npy"])
    st.markdown("</div>", unsafe_allow_html=True)
# uploaded_file = st.file_uploader("Upload your noisy signal (.h5 or .npy)", type=["h5", "npy"])

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
    
    .techno-head {
        font-family: 'Orbitron', sans-serif;
        color: #43fff1;
        font-size: 40px;
        text-align: center;
        margin-top: 0.5px;
        margin-bottom: 10px;
    }

    .techno-subhead {
        font-family: 'Orbitron', sans-serif;
        color: #43d7ff;
        font-size: 29px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Wider container */
    .main .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 80%;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare the signal
def load_signal(file):
    if file.name.endswith(".h5"):
        with h5py.File(file, 'r') as f:
            data = list(f.values())[0][()]  # first dataset
    else:
        data = np.load(file)

    # Ensure signal is 2D with shape (1024, 2)
    if data.ndim == 1:
        if data.shape[0] == 2048:
            data = data.reshape(1024, 2)
        else:
            raise ValueError(f"Expected 2048 values to reshape into (1024, 2), got {data.shape[0]}")
    elif data.shape != (1024, 2):
        raise ValueError(f"Expected shape (1024, 2), but got {data.shape}")
    
    return data

def preprocess_signal(signal_2d):
    """
    Takes input of shape (1024, 2) and returns tensor (1, 2, 1024)
    """
    signal = signal_2d.T  # to (2, 1024)
    signal = (signal - signal.mean()) / signal.std()
    signal = np.clip(signal, -1, 1)
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # (1, 2, 1024)
    return tensor

def denoise_signal(signal_tensor):
    with torch.no_grad():
        denoised = model(signal_tensor).squeeze().numpy()  # (2, 1024)
    return denoised.T  # Return to (1024, 2) for comparison/plotting

# def plot_signal(signal, title):
#     fig, ax = plt.subplots()
#     ax.plot(signal[:, 0], label='I (In-phase)')
#     ax.plot(signal[:, 1], label='Q (Quadrature)')
#     ax.set_title(title)
#     ax.set_xlabel("Sample Index")
#     ax.set_ylabel("Amplitude")
#     ax.legend()
#     st.pyplot(fig)

def plot_signal(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data[:, 0], mode='lines', name='I', line=dict(color='#38b1ff')))
    fig.add_trace(go.Scatter(y=data[:, 1], mode='lines', name='Q', line=dict(color='#6843ff')))
    fig.update_layout(title=title, xaxis_title="Sample Index", yaxis_title="Amplitude", 
                      template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Main app
if uploaded_file:
    try:
        signal = load_signal(uploaded_file)
        tensor = preprocess_signal(signal)
        denoised = denoise_signal(tensor)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="techno-subhead">Original (Noisy) Signal</div>', unsafe_allow_html=True)
            plot_signal(signal, "")
            st.markdown("**Note: I = In-phase, Q = Quadrature**")

        with col2:
            st.markdown('<div class="techno-subhead">Denoised Signal</div>', unsafe_allow_html=True)
            plot_signal(denoised, "")
            st.markdown("**Note: I = In-phase, Q = Quadrature**")

        # Download buttons centered
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

        npy_bytes = BytesIO()
        np.save(npy_bytes, denoised.astype(np.float32))
        st.download_button("Download as .npy", data=npy_bytes.getvalue(),
                           file_name="denoised_output.npy", mime="application/octet-stream")

        h5_bytes = BytesIO()
        with h5py.File(h5_bytes, 'w') as f:
            f.create_dataset("denoised_signal", data=denoised.astype(np.float32))
        h5_bytes.seek(0)
        st.download_button("Download as .h5", data=h5_bytes,
                           file_name="denoised_output.h5", mime="application/octet-stream")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
