import pandas as pd
import streamlit as st
import numpy as np
import librosa
import plotly.express as px
import plotly.graph_objects as go
from src.assignement_03.processor import STFT_manual


st.set_page_config(page_title="Signal frequency domain analysis", layout="wide")
st.title("Time-Frequency Analysis (STFT)")

st.markdown("""
### Song Selection

We selected two songs with very different characteristics : 
            
**Aksomaniac – Paapam** is a beat-driven track with strong bass and rapid changes over time.  
**Nuvole Bianche – Ludovico Einaudi** is a calm piano piece with smooth and stable frequency patterns.  

This contrast helps us clearly observe how different types of audio behave in the time-frequency domain using STFT.
""")

audio_option = st.selectbox("Select Audio", ["Paapam", "Nuvole Bianche"])

if audio_option == "Paapam":
    path = "data/assignement_03/paapam.wav"
else:
    path = "data/assignement_03/nuvole_bianche.wav"

total_duration = librosa.get_duration(path=path)

start_point = total_duration * 0.0

audio_duration = 180.0

signal, sr = librosa.load(path, sr=16_000, offset=start_point, duration=audio_duration)

st.audio(path, start_time=int(start_point))

tidy_audio_info = pd.DataFrame(
    {
        "Start Time [s]": [round(start_point, 2)],
        "End Time [s]": [round(start_point + audio_duration, 2)],
        "Duration [s]": [round(len(signal) / sr, 2)],
        "Sample Rate [Hz]": [sr],
        "Total Samples": [len(signal)],
        "Nyquist [Hz]": [sr / 2],
    },
    index=[audio_option],
)

st.write("### Audio Metadata")
st.dataframe(tidy_audio_info)

st.info(
    f"We decided to reduce the length of the signal because the sample size would be way to great. We reduced the time to {audio_duration}sec.",
    icon=":material/info:",
)

time_axis = np.linspace(0, len(signal) / sr, num=len(signal))

df_signal = pd.DataFrame({"Time [s]": time_axis, "Amplitude": signal})

st.write("### Time Domain Signal")

fig = px.line(
    df_signal,
    x="Time [s]",
    y="Amplitude",
    title=f"Signal of the {audio_duration}s Segment (Start: {start_point:.2f}s)",
    labels={"Time [s]": "Time (seconds)", "Amplitude": "Amplitude"},
    template="plotly_white",
)

fig.update_traces(line=dict(width=1, color="#1f77b4"))
fig.update_layout(
    hovermode="x unified",
    dragmode="zoom",
)

st.plotly_chart(fig, width="stretch")


st.header("1. Introduction")

st.markdown("""
In order to understand when a **Short-Time Fourier Transform (STFT)** is needed, 
one needs to understand what the DFT lacks. 

The DFT is great at analyzing the underlying frequencies and the general transformation 
from the time domain to the frequency domain. However, the information as to **WHEN** a certain frequency occurs in a signal gets lost. 

This is exactly where the **STFT shines**:
* It analyzes small **batches/windows** of the signal.
* It applies a **DFT** to each of these segments.
* This results in a **temporal frequency spectrum** (a spectrum over time).
""")

st.divider()

st.subheader("The Resolution Trade-off")

st.markdown("""
Naturally, one would ask: *What are good or optimal window sizes?* Unfortunately, there is a direct trade-off between the time and frequency resolution 
in the spectrum of the STFT. An **optimum** can only be achieved on a **case-by-case basis**, 
meaning there is no general global optimum.
""")

st.table(
    {
        "Focus": ["Time Resolution", "Frequency Resolution"],
        "Result": ["Lower Frequency Resolution", "Lower Time Resolution"],
        "Window Size": ["Small", "Large"],
    }
)

st.info(
    "**In other words: The higher the time resolution, the lower the frequency resolution and **vice versa**.**",
    icon=":material/info:",
)

st.subheader("The Window function")

st.markdown("""
Window functions are functions that are applied to each window during the STFT. They are applied to the window and everywhere else they are 0. This can actually be compared to an indicator function.
To perform an STFT, we must chop a long audio signal into short segments called "frames." Window functions are mathematical shapes applied to each of these frames to "soften the edges" of the cut.
            
There are many different window functions:
* Recantuglar
* Hamming
* Hann
* Blackman
* etc.
            
All these function differ in how signal is captured under window function. To understand
the difference between the functions, looking at the difference of the Rectangular and the others (like Hann) is most intuitive:
            
The difference between a Rectangular and a Hann window is essentially "hard" versus "soft" cutting. A Rectangular window is a simple hard cut that keeps the signal at full strength until it suddenly drops to zero at the edge. While it provides very narrow and sharp frequency peaks, it produces massive spectral leakage, making it difficult to measure quiet signals next to loud ones. In contrast, a Hann window (or Hanning window) uses a smooth, bell-shaped curve to fade the edges to zero.
""")

st.header("2. Methods")

st.markdown("""
To analyze the songs we looked at the STFT and the produced spectra with different parameters. We looked at how the spectra change when applying different window functions, window lengths, overlaps and also the scaling. Furthermore, we investigated how the energy changes for the frames and looked into different energies of different spans of frequencies to 
be able to deduce whether the singal is dominated by low, mid or high frequencies.

Last but not lest, we took a look at a further signal feature, the 'spectral centroid'. With the help of this feature we can then deduce if the signal sounds rather bright or dull and at which time frames.
""")

#################################################
# region: Spectra visualization
#################################################

st.header("3. Results")

st.subheader("The Window Function, Window Length, Overlaping and Scaling")

# STFT selection:
##################################

window_type_label = st.selectbox(
    "Window type", ["Rectangular", "Hamming", "Hann", "Blackman"]
)

col1, col2 = st.columns(2)

with col1:
    window_length = st.slider("Window length N", 128, 8192, 1024, step=128)

with col2:
    overlap_length = st.slider("Overlap length M", 0, window_length - 1, step=2)

scale_type_linear, scale_type_decible, scale_type_mel_scale = (
    "Linear (Magnitude)",
    "Decible (dB)",
    "Mel-Scale",
)

scale_type = st.radio(
    "Choose a scaling:",
    options=[scale_type_linear, scale_type_decible, scale_type_mel_scale],
    horizontal=True,
)

# STFT calculation:
##################################

stft_data = STFT_manual(
    signal,
    window_func_ind=window_type_label,
    window_size=window_length,
    overlap_size=overlap_length,
)

# STFT displaying:
##################################

stft_val = None

if scale_type == scale_type_linear:
    stft_val = stft_data

    fig = px.imshow(
        stft_data,
        origin="lower",
        aspect="auto",
        labels={"x": "Time (Frames)", "y": "Frequency (Bins)"},
        color_continuous_scale="Plasma",
        title="Magnitude",
    )

if scale_type == scale_type_decible:
    eps = 1e-09
    stft_db = 20 * np.log10(stft_data + eps)

    stft_val = stft_db

    fig = px.imshow(
        stft_db,
        origin="lower",
        aspect="auto",
        labels={"x": "Time (Frames)", "y": "Frequency (Bins)"},
        color_continuous_scale="Magma",
        title="DB-Spektrogramm",
    )

if scale_type == scale_type_mel_scale:
    S_mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=window_length,
        hop_length=window_length - overlap_length,
        n_mels=128,
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    stft_val = S_mel_db

    fig = px.imshow(
        S_mel_db,
        origin="lower",
        aspect="auto",
        labels={"x": "Time (Frames)", "y": "Frequency (Bins)"},
        color_continuous_scale="Magma",
        title="Mel-Spektrogramm",
    )

st.plotly_chart(fig)


fig_waterfall = go.Figure(
    data=go.Surface(
        x=np.arange(stft_data.shape[1]), y=np.arange(stft_data.shape[0]), z=stft_data
    )
)

fig_waterfall.update_layout(
    title="STFT Waterfall (Surface)",
    scene=dict(
        xaxis_title="Time (Frames)",
        yaxis_title="Frequency (Bins)",
        zaxis_title="Magnitude",
    ),
)
st.plotly_chart(fig_waterfall, width="stretch")


# endregion


#################################################
# region: Inspection of the Spectra
#################################################

num_bins, num_frames = stft_data.shape

max_freq = sr / 2
freq_step = max_freq / (num_bins - 1)

duration = len(signal) / sr

hop_size = window_length - overlap_length
time_step_sec = hop_size / sr

col1, col2 = st.columns(2)

with col1:
    st.metric("Frequency Range", f"0 - {max_freq} Hz")
    st.caption(f"Resolution: {freq_step:.2f} Hz per Bin")

with col2:
    st.metric("Time Range", f"0 - {duration:.2f} s")
    st.caption(f"{num_frames} Frames (Steps: {time_step_sec * 1000:.1f} ms)")

c1, c2 = st.columns(2)
with c1:
    inspect_bin = st.number_input("Bin Index:", 0, num_bins - 1, min(40, num_bins - 1))
with c2:
    inspect_frame = st.number_input(
        "Frame Index:", 0, num_frames - 1, min(60, num_frames - 1)
    )

f_start = inspect_bin * freq_step
f_end = f_start + freq_step
t_start = (inspect_frame * hop_size) / sr
t_end = t_start + (window_length / sr)

f_range = f"{f_start:.2f} - {f_end:.2f}"
t_range = f"{t_start:.3f} - {t_end:.3f}"

tidy_inspector = pd.DataFrame(
    {
        "Frequency range (Hz)": [f_range],
        "Time range (s)": [t_range],
        scale_type: [
            stft_data[inspect_bin, inspect_frame]
            if scale_type == scale_type_linear
            else stft_db[inspect_bin, inspect_frame]
            if scale_type == scale_type_decible
            else S_mel_db[inspect_bin, inspect_frame]
        ],
    },
    index=[f"{inspect_bin} / {inspect_frame}"],
)

st.dataframe(tidy_inspector)


# endregion


# st.dataframe(stft_val)


df = pd.DataFrame(stft_val)

# print(df)

power_stft = np.abs(stft_data) ** 2

mean_power = pd.DataFrame(power_stft).mean(axis=1)

# print(pd.DataFrame(power_stft))

fig = px.line(
    x=mean_power.index,
    y=mean_power.values,
    labels={"x": "Frequency Bin", "y": "Mean Power"},
    title="Mean Power-Spectrum across all frames",
)

st.plotly_chart(fig)

top_3_bins = mean_power.sort_values(ascending=False).head(50)

top3_bins_df = pd.DataFrame(
    {
        "value": top_3_bins.values,
        "from_hz": freq_step * top_3_bins.index,
        "to_hz": (freq_step * top_3_bins.index) + freq_step,
    },
    index=top_3_bins.index,
)

styled_df = (
    top3_bins_df.style.background_gradient(subset=["value"], cmap="magma")
    .format({"value": "{:.2f}", "from_hz": "{:.1f} Hz", "to_hz": "{:.1f} Hz"})
    .set_caption("Top 3 Frequenz-Bins und deren spektraler Span")
)


st.dataframe(styled_df)

cumulative_percentage = (np.cumsum(mean_power) / np.sum(mean_power)) * 100

df = pd.DataFrame(
    {
        "Frequency Bin": np.arange(len(cumulative_percentage)),
        "Cumulative Power (%)": cumulative_percentage,
        "Absolute Power": mean_power,
    }
)

fig = px.line(
    df,
    x="Frequency Bin",
    y="Cumulative Power (%)",
    title="Cumulative Spectral Power Analysis (CDF)",
    hover_data=["Absolute Power"],
    template="plotly_dark",
)

cutoff = 92

cutoff_idx = np.where(cumulative_percentage >= cutoff)[0][0]

fig.update_layout(
    xaxis_title="Frequency Bin Index",
    yaxis_title="Percentage of Total Energy (%)",
    hovermode="x unified",
)

st.plotly_chart(fig)

st.subheader("Subband Energy Analysis")

power = np.abs(stft_data) ** 2

num_bins, num_frames = power.shape

# Define bands (4 bands example)
bands = {
    "Low": (0, int(num_bins * 0.1)),
    "Low-Mid": (int(num_bins * 0.1), int(num_bins * 0.3)),
    "High-Mid": (int(num_bins * 0.3), int(num_bins * 0.6)),
    "High": (int(num_bins * 0.6), num_bins),
}

st.markdown(" Here the each band is defined as follows: ")

for band_name, (start, end) in bands.items():
    st.write(
        f"**{band_name}**: Bins {start} to {end} (Frequency range: {start * freq_step:.1f} Hz - {end * freq_step:.1f} Hz)"
    )

band_energy_per_frame = {}

for band_name, (start, end) in bands.items():
    band_energy = np.sum(power[start:end, :], axis=0)
    band_energy_per_frame[band_name] = band_energy

# Normalize per frame
total_energy_per_frame = np.sum(power, axis=0)

relative_band_energy = {
    band: energy / (total_energy_per_frame + 1e-9)
    for band, energy in band_energy_per_frame.items()
}


eqn = " Energy  = magnitude^2 "
st.latex(eqn)
st.write(
    "This means that the energy of a signal at a particular frequency bin is proportional to the square of the magnitude of the STFT at that bin. This relationship is crucial for understanding how much of the signal's energy is concentrated in different frequency bands, which is essential for tasks like audio analysis, feature extraction, and signal processing."
)

eqn1 = " power = |STFT|^2 "
st.latex(eqn1)
st.write(
    "Here, the power of the signal at each frequency bin is calculated by taking the magnitude of the STFT and squaring it. This gives us a measure of how much energy is present in each frequency bin, which can be used for further analysis such as identifying dominant frequencies, calculating spectral features, or performing tasks like noise reduction and audio classification."
)


eqn2 = " relative band energy = band energy / total energy per frame "
st.latex(eqn2)
st.write(
    "This equation represents the normalization of the energy in each frequency band by the total energy across all frequency bins for each time frame. By dividing the energy of a specific band by the total energy in that frame"
)

df_band = pd.DataFrame(relative_band_energy)

fig = px.line(
    df_band,
    title="Relative Band Energy per Frame",
    labels={"value": "Energy", "index": "Frame"},
)

st.plotly_chart(fig)


st.write(
    "The number of frames is the same as the number of columns in the STFT, which corresponds to the time dimension. Each point on the x-axis represents a specific time frame, and the y-axis shows the relative energy for each band at that time frame. This visualization allows us to see how the energy distribution across different frequency bands changes over time, providing insights into the temporal dynamics of the audio signal."
)

st.write(
    "This visualization shows that the sound is very deep, with 89% (song: Papam) / 93%(song: Nuvole Bianche) of its energy staying in the Low frequency band. The jumping lines in the graph mean that while the bass is always strong, there are quick moments where sharp or high noises briefly pop up, changing the sound's balance for a split second."
)

# the count of frames each band energy is observed in a data frame


# i want to combine both the count of frames and the average energy in a data frame to compare them side by side
frame_count = {
    band: np.sum(energy > 0.01)  # Count frames where energy is above a threshold
    for band, energy in relative_band_energy.items()
}


avg_energy = {band: np.mean(values) for band, values in relative_band_energy.items()}

df_band_stats = pd.DataFrame(
    {"Frame Count (>0.01)": frame_count, "Average Relative Energy": avg_energy}
)

st.write("### Band Energy Statistics")
st.dataframe(df_band_stats)


st.subheader("Additional Feature: Spectral Centroid")

freqs = np.linspace(0, sr / 2, num_bins)

spectral_centroid = np.sum(freqs[:, None] * power, axis=0) / (
    np.sum(power, axis=0) + 1e-9
)

fig = px.line(
    y=spectral_centroid,
    title="Spectral Centroid over Time",
    labels={"x": "Frame", "y": "Frequency (Hz)"},
)

st.plotly_chart(fig)


st.write(
    "We analyzed the sound by looking at different frequency groups. As already stated, our results show that the recording is very deep and bass-heavy, because about 90% of the total energy is found in the 'Low' band. We also added the Spectral Centroid as a special metric. This metric finds the 'middle point' of the sound. Because this point stays very low on the scale, it proves mathematically that the sound is mostly made of low tones. The spikes in our graph show the short moments when the sound became higher or sharper."
)

st.title("4. Discussion")

st.markdown("""

### Key Findings   

            
* **STFT** provides both time and frequency information  
* The choice of **window parameters** (type, length, overlap) significantly affects the spectrogram and the trade-off between time and frequency resolution  
* The **energy distribution across frequency bands** reveals key characteristics of the audio signal (e.g., dominance of low, mid, or high frequencies)  
* The **spectral centroid** indicates the “brightness” of a sound (lower = darker, higher = brighter)  
* There are **clear differences between rhythmic and harmonic signals**

We found it interesting that the scales are not all equally relevant for our specific use case. For the songs we chose, the **Magnitude scale** is not very revealing. However, the **Decibel (dB)** and **Mel scales** provide a much better picture of the underlying frequencies over time.
 
These two are also more useful in general when it comes to **human perception**. All of them have a connection to the magnitude scale and are therefore always related to the energy in the frequency. To be more precise, when we compare the scales:
 
* **Magnitude:** Represents the raw physical energy.
* **Decibel (dB):** It compresses the dynamic range, making it easier to see small, quiet details.
* **Mel Scale:** It is even more related to human perception and pitch; for example, a difference between 500 Hz and 600 Hz is clearly noticeable, but a difference with the same delta in the 10 kHz range is not.
 
We observed in our signal the typical **trade-off between time and frequency resolution** expected when dealing with STFTs. We found that a higher window length increases the granularity of the frequencies per frequency bin. This change significantly impacts the resulting spectra, providing better frequency detail at the cost of time precision.
 
To think a step further, when it comes to **classification problems** in this domain—for example, trying to classify different songs based on given features—we would:
 
* **Band Energy:** Analyze the distribution across all frames to identify the "tonal weight."
* **Spectral Centroid:** Use the average "brightness" across all frames as a signature.
""")