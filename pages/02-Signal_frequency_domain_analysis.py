import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.assignement_02.processor import (
    DFT_batched_downsampled,
    downsample_signal,
    get_sampling_rate,
    DFT_batched,
)

st.set_page_config(page_title="Signal frequency domain analysis", layout="wide")

st.title("Signal frequency domain analysis")


@st.cache_data
def load_signal():
    df = pd.read_csv("./data/assignement_02/raw/signal.csv")
    return df


@st.cache_data
def load_event():
    df = pd.read_csv("./data/assignement_02/raw/events.csv")
    return df


df = load_signal()
sampling_rate = get_sampling_rate(df)

fig = px.line(
    df,
    x="time_s",
    y="signal",
    labels={"time_s": "Time (sec.)", "signal": "Signal"},
)
fig.update_traces(opacity=0.8, line_width=0.5)
fig.update_layout(title="Signal")
st.plotly_chart(fig, use_container_width=True)

st.header("1. Introduction")

st.write(
    """
    The main difference between the DFT and the iDFT lies within the direction of transformation. If we want to transform something
    from the time domain to the frequency domain we need to use the DFT, but if we want to do the reverse, so from frequency domain to time domain,
    we have to take the inverse DFT. What the DFT basically does is that it takes any signal and 
    tries to decompose it into sinusoids and cosinusoids, with different frequencies and calculates how much of these components are essentially in the signal. 
    """
)

st.markdown(
    """
    Some important properties of the DFT:
    * The range of the frequencies is always related to the sampling rate. The range is always -fs/2 - fs/2.
    So if we have a sampling rate e.g. of **fs = 100Hz** with sampling duration **L = 10sec**, we would have a range of **-50Hz** to **+50Hz**. This means we have a total of **1,000 samples (10s⋅100Hz)** that are distributed across this frequency range, giving us exactly **1,000** frequency bins. This pretty much means,
    that there is a direct correlation between the amount of frequency bins/resolution and the length of the signal. In other words, the higher the duration of the signal, the higher the resolution we can capture.
    In the example above, we can for example see, that we can capture a resolution of 0.1Hz (100Hz / 1,000samples), but would we have 10,000 samples we could even capture a resolution of 0.01 Hz.
    * The Magnitude Spectrum represents the actual amplitude (the physical height) of every individual sine wave found within your signal. 
    In contrast, the Power Spectrum is calculated by squaring the magnitude to emphasize the energy contained in the signal. This is particularly useful for separating dominant signals from background noise.
    """
)

st.header("2. Methods")

st.write(
    """
    We invesigated the signal by trying to apply a manually implemented DFT, and plot the resulting magnitude and power spectrum. Based on these spectra we were able to filter out the meaningful and relevant
    frequencies in our spectrum. With these methods we were able to figure out which frequency components are actual signal and which ones are just noise.
    Furthermore, we looked into downsampling and observed its effects on our signal.
    """
)

st.header("3. Results")

st.markdown(f"""
### The signal at hand, has a sampling rate of: {sampling_rate} Hz
""")

st.write(
    f"""
Looking at this sampling rate in respect to Nyquist Criterion, it can be deduced, that the highest frequency should not be 
higher than {sampling_rate / 2} Hz. Otherwise, we would have aliasing effects.
"""
)

st.write(
    """
    To implement a DFT (dsicrete fourier transform) by hand is not hard. It only gets tricky when there is a huge amount of data, because even though 
    it is possible to implement the DFT efficiently, by working with matrices (instead of double loops), the sheer amount of data does not
    make it possible for us to implement a DFT version that is capable to calculate everything (whole recording) at once:
    """
)

code = """def DFT(signal):
  N = len(signal)
  n = np.arange(N)
  k = n.reshape((N, 1))
  e = np.exp(-2j * np.pi * k * n / N)
  dft = np.dot(e,signal)
  return dft

dft = DFT(singal_df["signal"].values)"""
st.code(code, language="python")

st.write(
    """
    The reason for this is limitation is the time-complexity (including used memory) of this algorithm:
    """
)
st.latex(r"\text{Complexity: } \mathcal{O}(N^2)")

st.warning(
    f"""  ⚠️ This results in approximately **{460800**2:_}** (over 212 B) operations for our dataset. 
    Processing this at once exceeds the available RAM by many magnitudes and results in a kernel crash."""
)

st.markdown("""
    # DFT of signal""")

st.write(
    """
    Therefor we decided to investigate only smaller segments of the data.
    """
)

values = st.slider("Select a range of values", 0, 1800, (0, 30), step=1)


def plot_restricted_signal(start, end):
    fig = px.line(
        df.loc[(df["time_s"] >= start) & (df["time_s"] <= end), :],
        x="time_s",
        y="signal",
        labels={"time_s": "Time (sec.)", "signal": "Signal"},
    )
    fig.update_traces(opacity=0.8, line_width=0.5)
    fig.update_layout(title=f"Signal - {start} sec. : {end} sec.")
    st.plotly_chart(fig, use_container_width=True)


plot_restricted_signal(values[0], values[1])

amp, freq = DFT_batched(signal=df, start=values[0], end=values[1])

diff = values[1] - values[0]

if diff <= 10:
    st.caption(f"✅ Ideal range for DFT ({diff}s)")
elif diff <= 30:
    st.caption(f"⚠️ Warning: Calculation might take a few seconds ({diff}s)")
else:
    st.caption(f"❌ Critical: {diff}s is too much for $O(N^2)$!")

magnitude_df = pd.DataFrame({"freq_hz": freq[1:], "amplitude": amp[1:]})

fig_spec = px.line(
    magnitude_df,
    x="freq_hz",
    y="amplitude",
    labels={"freq_hz": "Frequency (Hz)", "amplitude": "Amplitude"},
)

fig_spec.update_traces(opacity=0.8, line_width=1.0)
fig_spec.update_layout(
    title=f"Magnitude Spectrum - {values[0]} sec. : {values[1]} sec."
)
fig_spec.add_vline(
    x=128, line_width=1.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.5)"
)

st.plotly_chart(fig_spec, use_container_width=True)

st.markdown("""
### 
We marked the **Nyquist-Frequency** in the spectrum with a dotted line. 

* **Symmetry:** Left and right side are always equivalent.
* **Range:** The actual frequency ranges from **-128Hz to +128Hz**. 

**Note:** In theory, everything past 128Hz should be removed and copied to the left side, so that for example the frequency of **156Hz** would exactly land at **-100Hz**.
""")

st.markdown(
    """
    The following table shows only the most prevelant frequencies that are **NOT extremely low (close to 0 Hz)** (excluded), but also
    can be seen as peaks in the spectrum (only in range: 0 - fs/2):
    """
)

st.dataframe(
    magnitude_df.loc[
        (magnitude_df["freq_hz"] > 10) & (magnitude_df["freq_hz"] < 128), :
    ]
    .sort_values(by=["amplitude"], ascending=False)
    .head(4)
)

st.markdown("""
    # Signal vs. Noise""")

st.write(
    """
    However, even though the extremly low frequencies are not included in the table above, we think they are still really relevant.
    To investigate further what role they play, we decided to look into the power spectrum of the whole signal, so we
    be more sure what is actual signal and what is just noise or interference.
    """
)

amp, freq = DFT_batched(signal=df, start=values[0], end=values[1])

amp = amp**2

magnitude_df = pd.DataFrame({"freq_hz": freq, "power": amp})

fig_spec = px.line(
    magnitude_df,
    x="freq_hz",
    y="power",
    labels={"freq_hz": "Frequency (Hz)", "power": "Power"},
)

fig_spec.update_traces(opacity=0.8, line_width=1.0)
fig_spec.update_layout(
    title={
        "text": "Power Spectrum",
        "subtitle": {"text": "Includes a symmetrical offset of 100."},
        "x": 0.05,
        "xanchor": "left",
    }
)
fig_spec.add_vline(
    x=128, line_width=1.5, line_dash="dash", line_color="rgba(205, 92, 92, 1.0)"
)

st.plotly_chart(fig_spec, use_container_width=True)

st.write(
    """
    This power spectrum clearly distinguishes the actual signal from the noise. 
    Most of the energy is concentrated at very low frequencies (close to 0 Hz), confirming that the primary signal consists of slow trends and shifts.
    The sharp peaks at higher frequencies are identified as noise, possibly from electrical interference.
    This low-frequency component also corresponds to the DC offset of 1.0 observed in the time domain.
    """
)

st.markdown("""
    # Event markers""")

st.write(
    """
    As a data source we not only had the signal to analyze, but also event markers. These event markers in time domain and their distribution
    is displayed below:
    """
)

df_events = load_event()
df_signal = load_signal()

fig_2 = px.line(
    df,
    x="time_s",
    y="signal",
    labels={"time_s": "Time (sec.)", "signal": "Signal"},
)
fig_2.update_traces(opacity=0.8, line_width=0.5)
fig_2.update_layout(title="Signal")

for marker_pos in df_events["marker_time_s"]:
    fig_2.add_vline(
        x=marker_pos,
        line_width=1.5,
        line_dash="dash",
        line_color="rgba(205, 92, 92, 1.0)",
    )

st.plotly_chart(fig_2, use_container_width=True)

df_events["marker_diff"] = (
    df_events["marker_time_s"].shift(-1).iloc[:-1]
    - df_events.iloc[:-1]["marker_time_s"]
)

fig_03 = px.histogram(
    x=df_events[:-1]["marker_diff"],
    marginal="box",
    histnorm="probability density",
    nbins=50,
    title="Distribution of time difference of events",
    color_discrete_sequence=["indianred"],
)

fig_03.update_layout(xaxis_title="Time Difference [s]")

st.plotly_chart(fig_03, use_container_width=True)

st.markdown(
    """
    The event markers show a clear pattern happening about every 11 to 12 seconds, which proves they are part of the actual signal and not just random noise.
    Every time a marker appears, the signal drops sharply, showing a real physical reaction. 
    The high-frequency noise (the jitter) is just sitting on top of these drops.

    The distribution of the events suggests a physical trigger or a digital executed one (but NOT periodically). These triggers
    could be anything, like:
    * A trigger / sensor that responses to some event (e.g. in a warehouse).
    * Garbage collector (GC) cycle (dereferenced objects in RAM) executing on trigger.
    * Or something extremly simple, like keeping a button pressed for that time (amplitude is press intensity).
    * etc.
    """
)
st.markdown("""
    # Downsampling""")

st.write(
    """
    What happens with our magnitude and the power sepctrum when we try to downsample?
    """
)

signal_df = load_signal()
original_fs = get_sampling_rate(signal_df)

# Only allow safe values (power of 2)
sampling_options = [256, 128, 64, 32, 16, 8, 4, 2]

target_fs = st.select_slider(
    "Select Sampling Rate (Hz)", options=sampling_options, value=256
)


downsampled_df, actual_fs, factor = downsample_signal(signal_df, original_fs, target_fs)

st.write(f"Original Sampling Rate: **{original_fs:.2f} Hz**")
st.write(f"Downsampling Factor: **{factor}**")
st.write(f"New Sampling Rate: **{actual_fs:.2f} Hz**")

st.warning("⚠️ No filtering applied → Aliasing may occur!")


st.subheader("Time Domain Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("Original Signal")
    fig_orig = px.line(signal_df, x="time_s", y="signal")
    fig_orig.update_traces(line_width=0.5)
    st.plotly_chart(fig_orig, use_container_width=True, key="original_time")

with col2:
    st.write("Downsampled Signal")
    fig_ds = px.line(downsampled_df, x="time_s", y="signal")
    fig_ds.update_traces(line_width=0.5)
    st.plotly_chart(fig_ds, use_container_width=True, key="downsampled_time")

start, end = values

max_points = 2000  # safe limit
duration = end - start

if duration * actual_fs > max_points:
    duration = max_points / actual_fs
    end = start + duration
    st.warning("⚠️ Window auto-limited to avoid crash")


st.subheader("Frequency Domain Comparison")

amp_orig, freq_orig = DFT_batched(signal=signal_df, start=start, end=end)

amp_ds, freq_ds = DFT_batched_downsampled(
    signal=downsampled_df, start=start, end=end, fs=actual_fs
)

# ORIGINAL
df_orig_mag = pd.DataFrame({"freq": freq_orig, "value": amp_orig, "type": "Magnitude"})

df_orig_pow = pd.DataFrame({"freq": freq_orig, "value": amp_orig**2, "type": "Power"})

df_orig_all = pd.concat([df_orig_mag, df_orig_pow])

df_ds_mag = pd.DataFrame({"freq": freq_ds, "value": amp_ds, "type": "Magnitude"})

df_ds_pow = pd.DataFrame({"freq": freq_ds, "value": amp_ds**2, "type": "Power"})

df_ds_all = pd.concat([df_ds_mag, df_ds_pow])

col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(
        df_orig_all,
        x="freq",
        y="value",
        color="type",
        title="Original Spectrum",
    )

    fig1.add_vline(
        x=original_fs / 2, line_dash="dash", line_color="rgba(205, 92, 92, 1.0)"
    )
    fig1.update_yaxes(range=[0, 20000])
    st.plotly_chart(fig1, use_container_width=True, key="orig_spec")

with col2:
    fig2 = px.line(
        df_ds_all,
        x="freq",
        y="value",
        color="type",
        title="Downsampled Spectrum (Aliasing visible here)",
    )

    fig2.add_vline(
        x=actual_fs / 2, line_dash="dash", line_color="rgba(205, 92, 92, 1.0)"
    )
    fig2.update_yaxes(range=[0, 5000])
    st.plotly_chart(fig2, use_container_width=True, key="ds_spec")

st.write(
    """
    The alising effect can clearly be seen when downsampling from **fs=256 to a fs=64** where the frequencies get pushed "across the edge" (due to the assumption of periodicity of DFT).
    The problem of aliasing is that we would misclassify a signal component as being prevelant, even though it is not and just occurs because it got pushed to the otherside.
    """
)

st.header("4. Discussion")

st.write(
    """
    Our signal has some high frequency noise, especially at higher level frequencies (that represent the jitter), there is some noise in the signal.
    To be precise, the noise frequencies we identified, can be seen in table we have above. These noise frequencise are at around 16.6, 33.3, 50, 100. We
    think that these noise signals could come from some electrical interference. Since the power frequency in Europe is exactly 50Hz and at exactly 50Hz we have
    a huge noise component, we think that the measurment was probably done in europe with an electronic device. In North-America we would therefor expect a peak at 60HZ for the same measurment.
    """
)

st.write(
    """
    We found that it is not really straightforward to conclude what is noise and what is the actual signal, without
    knowing something about its origin. We were struggeling to really know if the peaks are really noise or if they are
    the actual, underlying signal.
    """
)
