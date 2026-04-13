import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="F1 Telemetry Analysis", layout="wide")

st.title(" Time Domain Analysis of F1 Telemetry Data")


@st.cache_data
def load_data():
    df = pd.read_csv("./data/assignement_01/processed/preprocessed_VER.csv")
    return df


df = load_data()
df = df.sort_values(by=["LapNumber", "Time_ms"])


st.image("./resources/Bahrain_Circuit.avif", caption="Bahrain Circuit")

st.markdown("""
### The bahrain Circuit is a 5.412 km track with 15 turns and 3 drs zones, known for its mix of long straights and technical corners.
""")

st.header("1. Introduction")

st.write(
    "When we look at these telemetry variables, we are dealing with a signal in the time domain. "
    "The x-axis represents time, in our case milliseconds after the start of the lap, and the y-axis represents the amplitude of the signal, which here is one of the three variables we chose. "
    "So at any given moment in time, we e.g. have one speed value, and the full lap is represented as one continuous curve over roughly 95 seconds."
)

st.write(
    "What makes the time domain useful is that we can directly read off events from the signal. "
    "For example, when the speed drops sharply, we know the driver is braking. When it reaches a local minimum, the car is at minimum speed. "
    "When it rises again, the driver is accelerating out. This makes it very natural to identify patterns, trends, and events just by looking at the shape of the curve."
)

st.header("2. Method")

st.write(
    "First we had to preprocess the data accordingly, so we can actually make visualisations. For that "
    "we mainly had to focus on getting the times right. First the times in the df were cumulative over all laps, so "
    "we reset the time after each lap to zero and calculated the cumsum over each lap."
)

st.write(
    "To analyze the data we did two things, first we looked at basic statistical measures over a lap, like "
    "the average pace of finishing a lap or the standard deviation across all laps, to get a first intuition on "
    "how consistent the driver is. For Verstappen we found that he is really consistent, with a small std of only around 4 sec. and a "
    "median pace of around 95 sec."
)

df_raw = load_data()

grps = df_raw.groupby("LapNumber")

results = []

for lap_no, group in grps:
    last_row = group[["LapNumber", "Time_ms"]].iloc[-1]
    results.append(last_row)

df_lap_times = pd.DataFrame(results).reset_index(drop=True)

df_lap_times["Time_ms"] = df_lap_times["Time_ms"].astype("float")

df_lap_times = df_lap_times.sort_values(by="Time_ms", ascending=True)

col1, col2, col3 = st.columns(3)
col1.metric("Best Lap", f"{df_lap_times['Time_ms'].iloc[0] / 1000:.3f}s")
col2.metric("Mean Pace", f"{df_lap_times['Time_ms'].mean() / 1000:.3f}s")
col3.metric("Standard deviation", f"{df_lap_times['Time_ms'].std() / 1000:.3f}s")

st.write(
    "At first we wanted to get an overview over the data in general, and see how the different "
    "telemetry variables evolve over time. To achieve that at first, we plotted everything into one "
    "single plot, but we recognized two issues: way different scales of the different variables "
    "and that the visulisation itself started to lag a lot."
)
st.write(
    "That is why we decided to create a plot for the different telemetry variables. Furthermore, "
    "we wanted to devided the laps in meaningful parts/charts, to avoid the lagging. For that we "
    "devided it into three phases, the opening, middle and closing phase. The openening and closing "
    "phases only consist of some laps (bulk of laps is in middle), because they are more eventful and "
    "probably contain more variability and special laps (like first and last laps)."
)

st.write(
    "Moreover to see statistical measures and how they behave over different time windows and over different laps "
    "we decided to create a further plot with adjustable parameters (Search Statical measurements according to the Lap and Time)."
    "With this approach we are not only constraint by one certain point in time, but can investigate a whole range of time. This"
    "makes the analysis easier and better to handle."
)

st.write(
    "Since we did not only want to create insight based on each variable in respect to the time-domain, but "
    "also look into how the variables correlate between each other, we plotted the correlation matrix to "
    "see how and if the variables are correlated. "
)

st.header("3. Results")


def plot_variable(data, var, title):
    fig = px.line(
        data,
        x="Time_ms",
        y=var,
        color="LapNumber",
        line_group="LapNumber",
        labels={
            "Time_ms": "Time (sec.)",
        },
    )
    fig.update_traces(opacity=0.8, line_width=1)
    fig.update_layout(title=title, legend_title="Lap")
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 150 * 1000, 10 * 1000)),
            ticktext=[
                f"{x / 1000} sec." for x in list(range(0, 150 * 1000, 10 * 1000))
            ],
            tickson="boundaries",
            ticklen=20,
        )
    )
    st.plotly_chart(fig, use_container_width=True)


opening_phase = df[df["LapNumber"] <= 15]
middle_phase = df[(df["LapNumber"] > 15) & (df["LapNumber"] <= 45)]
closing_phase = df[df["LapNumber"] > 45]

pitstop_phase = df[
    ((df["LapNumber"] >= 17) & (df["LapNumber"] <= 19))
    | ((df["LapNumber"] >= 37) & (df["LapNumber"] <= 39))
]


st.header("Opening Phase (First 15 Laps)")

col1, col2 = st.columns(2)

with col1:
    plot_variable(opening_phase, "Speed", "Speed vs Time")

with col2:
    plot_variable(opening_phase, "RPM", "RPM vs Time")

plot_variable(opening_phase, "nGear", "Gear vs Time")


st.header(" Middle Phase (Laps 16-45)")

col1, col2 = st.columns(2)

with col1:
    plot_variable(middle_phase, "Speed", "Speed vs Time")

with col2:
    plot_variable(middle_phase, "RPM", "RPM vs Time")

plot_variable(middle_phase, "nGear", "Gear vs Time")

st.write(
    "Looking at the charts (excluding pit stop laps and the lap before that), we can "
    "see that the driver is really consistent in the approach to the first corner and "
    "the curves are nearly identical across all laps. However, with continouing time, "
    "there is more variablity across all three variables."
)


st.header(" Closing Phase (Laps 46+)")

col1, col2 = st.columns(2)

with col1:
    plot_variable(closing_phase, "Speed", "Speed vs Time")

with col2:
    plot_variable(closing_phase, "RPM", "RPM vs Time")

plot_variable(closing_phase, "nGear", "Gear vs Time")

st.header("Pit Stop Analysis")

col1, col2 = st.columns(2)

with col1:
    plot_variable(pitstop_phase, "Speed", "Speed vs Time")

with col2:
    plot_variable(pitstop_phase, "RPM", "RPM vs Time")

plot_variable(pitstop_phase, "nGear", "Gear vs Time")

st.write(
    "After plotting everything we saw that there are rather unusual patterns. It can be argued "
    "that these patterns are probably the pit stops, because we have at nearly the same time at "
    "lap 18 and lap 38 no speed. What is interesting is that it seems that he was accelerating earlier and more constant in lap 38 "
    "and was therefore able to finish the lap overall one second faster (shift in time between lap 18 and lap 38)."
)

st.write(
    "For comparsion reasons we also plotted one lap before and one after the pit stop lap."
)

st.write(
    "Besides that, what is noticable is that in lap 19 he managed to accelerate faster again after exiting "
    "the curve and he was able to carry this projection to the end."
)

st.markdown("""
### Overall patterns
""")

st.markdown("""
 After comparing the patterns in the signals with the track, we are actually able to approximate
 where the corners are. Meaning, we can at least approximate when Verstappen breaks for a curve.
 However, we are aware that we can never be one hundred percent certain
 when he actually enters the corner and when he exists it, because we only look at the telemetry variables
 and for example he possibly could also break on a straight, and we would identify it as a corner. Furthermore,
 we would have to know where the actually corner starts and where it ends, which is information we also do not have.
""")


st.header(" Search Statical measurements according to the Lap and Time")

lap_range = st.slider(
    "Select Lap Range for Combined Analysis",
    int(df["LapNumber"].min()),
    int(df["LapNumber"].max()),
    (2, 10),
)

lap_time_range = st.slider(
    "Select Time Range for Combined Analysis (Sec)",
    int(df["Time_ms"].min() / 1000),
    int(df["Time_ms"].max() / 1000),
    (0, 150),
)
combined_df = df[
    (df["LapNumber"] >= lap_range[0])
    & (df["LapNumber"] <= lap_range[1])
    & (df["Time_ms"] >= lap_time_range[0] * 1000)
    & (df["Time_ms"] <= lap_time_range[1] * 1000)
]

combined_mean = combined_df[["Speed", "RPM", "nGear"]].mean()
combined_median = combined_df[["Speed", "RPM", "nGear"]].median()
combined_std = combined_df[["Speed", "RPM", "nGear"]].std()
combined_range = (
    combined_df[["Speed", "RPM", "nGear"]].max()
    - combined_df[["Speed", "RPM", "nGear"]].min()
)
combined_df = combined_df.copy()
combined_df["Acceleration"] = (
    combined_df["Speed"].diff() / combined_df["Time_ms"].diff()
)
combined_acc_mean = combined_df["Acceleration"].mean()
combined_acc_max = combined_df["Acceleration"].max()
combined_acc_min = combined_df["Acceleration"].min()


combined_summary_df = pd.DataFrame(
    {
        "Mean": combined_mean,
        "Median": combined_median,
        "Std": combined_std,
        "Range": combined_range,
    }
)
st.subheader(" Statistical Summary for Selected Lap and Time Range")
st.dataframe(combined_summary_df)
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=combined_df["Time_ms"], y=combined_df["Speed"], name="Speed")
)
fig.add_trace(go.Scatter(x=combined_df["Time_ms"], y=combined_df["RPM"], name="RPM"))
fig.add_trace(
    go.Scatter(
        x=combined_df["Time_ms"], y=combined_df["nGear"], name="Gear", yaxis="y2"
    )
)
fig.update_layout(
    title=f"Combined Analysis: Laps {lap_range[0]}-{lap_range[1]}, Time {lap_time_range[0]}-{lap_time_range[1]} sec",
    xaxis_title="Time (ms)",
    yaxis=dict(title="Speed / RPM"),
    yaxis2=dict(title="Gear", overlaying="y", side="right"),
)
fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 150 * 1000, 10 * 1000)),
        ticktext=[f"{x / 1000} sec." for x in list(range(0, 150 * 1000, 10 * 1000))],
        tickson="boundaries",
        ticklen=20,
    )
)
st.plotly_chart(fig, use_container_width=True)

st.write(
    "To analyze the consistency across laps we investigated the amount of overlap we have. "
    "The closer the points are to each other the more consistency there is. For example by looking "
    "at the speed, we can analye the speed consistency across laps by looking at the amount of overlap. "
    "It should be noted, that in high acceleration and breaking phases we have a high rate of change in speed "
    "and therefore even if the points of other laps are close to each other they vary quite a lot."
)

# combined acceleration text
acc_mean = combined_df["Acceleration"].mean()
acc_max = combined_df["Acceleration"].max()
acc_min = combined_df["Acceleration"].min()

st.markdown(f"""
###  Acceleration Analysis
- Mean Acceleration: {acc_mean:.2f} m/s²
- Max Acceleration: {acc_max:.2f} m/s²
- Min Acceleration: {acc_min:.2f} m/s²
""")

# it's plot
fig = px.line(
    combined_df,
    x="Time_ms",
    y="Acceleration",
    color="LapNumber",
    line_group="LapNumber",
    labels={"Time_ms": "Time (sec.)", "Acceleration": "Acceleration (m/s²)"},
    title="Combined Lap and Time Acceleration Analysis",
)
fig.update_traces(opacity=0.8, line_width=1)
fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 150 * 1000, 10 * 1000)),
        ticktext=[f"{x / 1000} sec." for x in list(range(0, 150 * 1000, 10 * 1000))],
        tickson="boundaries",
        ticklen=20,
    )
)
st.plotly_chart(fig, use_container_width=True)
# correlation analysis for combined lap and time range
combined_corr = combined_df[["Speed", "RPM", "nGear"]].corr()
fig = px.imshow(combined_corr, text_auto=True, title="Correlation Matrix")
st.plotly_chart(fig, use_container_width=True)


st.header("4. Dicussion")

st.write(
    "Overall, we think one can already get a good understanding about the data/signals, just by looking "
    "at how the variables evolve over time and by looking at the statistical measures. From these two approaches "
    "one can already deduce a vast amount of information, without having to dive extremley deep into "
    "more advanced approaches."
)

st.write(
    "We think, when it comes to consistency across laps of a variable, it would be good, if we could "
    "display this consistency not just by looking at the plot, but by also having a good representative measure, but this "
    "is not easy with such time data where we do not have the exact same time points across multiple laps, e.g "
    "we do not have a data record at exaclty 50ms after the start of the lap for all laps. "
    "To conquer this problem we could fix a certain laps times (e.g. lap 1: 50ms after start) and approximate "
    "by (linear) interpolation the values of the other laps (e.g. lap 2: by taking one timepoint below and above that and interpolate at 50ms). "
    "Such a measure would be really valuable and one generate a lot of information this way."
)
