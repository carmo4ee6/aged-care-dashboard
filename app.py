# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Residential Aged Care – Provider", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=2)

    # Clean numeric fields
    for col in ["Residential Places", "Home Care Places", "Restorative Care Places", "2024-25 Australian Government Funding"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardise strings
    for col in ["Physical State", "Care Type", "Provider Name", "Organisation Type", "ABS Remoteness",
                "2018 Aged Care Planning Region (ACPR)", "2016 SA3 Name", "2023 LGA Name", "2017 PHN Name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Keep only rows with coordinates for mapping
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    return df

def haversine_km(lat1, lon1, lat2, lon2):
    # Vectorised haversine (km)
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def format_money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}m"
    return f"${value:,.0f}"

def style_table(df: pd.DataFrame, numeric_columns=None, formatters=None):
    if numeric_columns is None:
        numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_columns = list(numeric_columns)
    text_columns = [c for c in df.columns if c not in numeric_columns]

    styler = df.style
    if formatters:
        styler = styler.format(formatters, na_rep="N/A")
    if numeric_columns:
        styler = styler.set_properties(subset=numeric_columns, **{"text-align": "right"})
    if text_columns:
        styler = styler.set_properties(subset=text_columns, **{"text-align": "left"})

    header_styles = []
    for idx, col in enumerate(df.columns):
        align = "right" if col in numeric_columns else "left"
        header_styles.append(
            {"selector": f"th.col_heading.level0.col{idx}", "props": [("text-align", align)]}
        )
    return styler.set_table_styles(header_styles, overwrite=False)

DATA_PATH = "Service-List-2025-Australia_300126.xlsx"
df = load_data(DATA_PATH)

st.title("Residential Aged Care: Provider & Competition Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

state = st.sidebar.multiselect(
    "State",
    options=sorted(df["Physical State"].dropna().unique())
)

care_type = st.sidebar.multiselect(
    "Care Type",
    options=sorted(df["Care Type"].dropna().unique()),
    default=["Residential"] if "Residential" in df["Care Type"].unique() else None
)

org_type = st.sidebar.multiselect(
    "Organisation Type",
    options=sorted(df["Organisation Type"].dropna().unique())
)

acpr = st.sidebar.multiselect(
    "ACPR",
    options=sorted(df["2018 Aged Care Planning Region (ACPR)"].dropna().unique())
)

provider_search = st.sidebar.text_input("Provider name contains (optional)", "")

dff = df.copy()
if state:
    dff = dff[dff["Physical State"].isin(state)]
if care_type:
    dff = dff[dff["Care Type"].isin(care_type)]
if org_type:
    dff = dff[dff["Organisation Type"].isin(org_type)]
if acpr:
    dff = dff[dff["2018 Aged Care Planning Region (ACPR)"].isin(acpr)]
if provider_search.strip():
    dff = dff[dff["Provider Name"].str.contains(provider_search.strip(), case=False, na=False)]

if dff.empty:
    st.warning("No rows match the current filters. Adjust filters to continue.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Facilities", f"{len(dff):,}")
c2.metric("Providers", f"{dff['Provider Name'].nunique():,}")
c3.metric("Residential places", f"{int(dff['Residential Places'].fillna(0).sum()):,}")
funding_sum = dff["2024-25 Australian Government Funding"].fillna(0).sum()
c4.metric("Govt funding (sum)", format_money(funding_sum))

places_sum = dff["Residential Places"].fillna(0).sum()
avg_funding_per_facility = funding_sum / len(dff) if len(dff) else np.nan
avg_funding_per_place = funding_sum / places_sum if places_sum else np.nan
median_facility_funding = dff["2024-25 Australian Government Funding"].median(skipna=True)

c5, c6, c7 = st.columns(3)
c5.metric(
    "Avg funding / facility",
    format_money(avg_funding_per_facility)
)
c6.metric(
    "Avg funding / place",
    format_money(avg_funding_per_place)
)
c7.metric(
    "Median facility funding",
    format_money(median_facility_funding)
)

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Map")
    map_df = dff.dropna(subset=["Latitude", "Longitude"]).copy()
    # Streamlit's built-in map uses lat/lon columns named exactly like this:
    st.map(map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"}))

with right:
    st.subheader("Top providers by residential places")
    top = (
        dff.groupby(["Provider Name", "Organisation Type"], dropna=False)["Residential Places"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_styler = style_table(
        top,
        numeric_columns=["Residential Places"],
        formatters={"Residential Places": "{:,.0f}"},
    )
    st.dataframe(top_styler, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Funding insights")

f_left, f_right = st.columns(2)

with f_left:
    st.markdown("**Funding by state**")
    funding_by_state = (
        dff.groupby("Physical State", dropna=False)["2024-25 Australian Government Funding"]
        .sum()
        .sort_values(ascending=False)
        .rename("Funding")
        .reset_index()
    )
    funding_by_state["Funding_m"] = funding_by_state["Funding"] / 1_000_000
    state_chart = (
        alt.Chart(funding_by_state)
        .mark_bar()
        .encode(
            x=alt.X("Physical State:N", sort="-y"),
            y=alt.Y("Funding_m:Q", axis=alt.Axis(format=",.1f", title="Funding ($m)")),
            tooltip=[
                alt.Tooltip("Physical State:N", title="State"),
                alt.Tooltip("Funding:Q", format="$,.0f"),
            ],
        )
    )
    st.altair_chart(state_chart, use_container_width=True)

with f_right:
    st.markdown("**Funding by organisation type**")
    funding_by_org = (
        dff.groupby("Organisation Type", dropna=False)["2024-25 Australian Government Funding"]
        .sum()
        .sort_values(ascending=False)
        .rename("Funding")
        .reset_index()
    )
    funding_by_org["Funding_m"] = funding_by_org["Funding"] / 1_000_000
    org_chart = (
        alt.Chart(funding_by_org)
        .mark_bar()
        .encode(
            x=alt.X("Organisation Type:N", sort="-y"),
            y=alt.Y("Funding_m:Q", axis=alt.Axis(format=",.1f", title="Funding ($m)")),
            tooltip=[
                alt.Tooltip("Organisation Type:N"),
                alt.Tooltip("Funding:Q", format="$,.0f"),
            ],
        )
    )
    st.altair_chart(org_chart, use_container_width=True)

funding_provider = (
    dff.groupby(["Provider Name", "Organisation Type"], dropna=False)[
        ["2024-25 Australian Government Funding", "Residential Places"]
    ]
    .sum()
    .sort_values("2024-25 Australian Government Funding", ascending=False)
    .head(20)
    .reset_index()
)
funding_provider["Funding per place"] = np.where(
    funding_provider["Residential Places"] > 0,
    funding_provider["2024-25 Australian Government Funding"] / funding_provider["Residential Places"],
    np.nan
)

st.markdown("**Top providers by funding**")
funding_provider_display = funding_provider[
    [
        "Provider Name",
        "Organisation Type",
        "2024-25 Australian Government Funding",
        "Residential Places",
        "Funding per place",
    ]
].copy()

funding_styler = style_table(
    funding_provider_display,
    numeric_columns=[
        "2024-25 Australian Government Funding",
        "Residential Places",
        "Funding per place",
    ],
    formatters={
        "2024-25 Australian Government Funding": lambda x: f"${np.round(x / 1000) * 1000:,.0f}" if pd.notna(x) else "N/A",
        "Residential Places": "{:,.0f}",
        "Funding per place": lambda x: f"${np.round(x / 1000) * 1000:,.0f}" if pd.notna(x) else "N/A",
    },
)

st.dataframe(
    funding_styler,
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Competition within radius (around a selected facility)")

facility_options = (
    dff[["Service Name", "Provider Name", "Physical Suburb", "Physical State", "Latitude", "Longitude"]]
    .dropna(subset=["Latitude", "Longitude"])
    .copy()
)
facility_options["label"] = (
    facility_options["Service Name"] + " — " +
    facility_options["Provider Name"] + " (" +
    facility_options["Physical Suburb"].astype(str) + ", " +
    facility_options["Physical State"].astype(str) + ")"
)

if facility_options.empty:
    st.info("No facilities with coordinates are available for the current filters.")
    st.stop()

selected = st.selectbox("Select a facility", options=facility_options["label"].tolist())
radius_km = st.slider("Radius (km)", min_value=1, max_value=80, value=10)

sel_row = facility_options[facility_options["label"] == selected].iloc[0]
lat0, lon0 = float(sel_row["Latitude"]), float(sel_row["Longitude"])

cand = dff.dropna(subset=["Latitude", "Longitude"]).copy()
cand["dist_km"] = haversine_km(lat0, lon0, cand["Latitude"].astype(float), cand["Longitude"].astype(float))
within = cand[cand["dist_km"] <= radius_km].copy()

# Basic competitor summary
# (If you want "exclude the selected facility's provider", filter here.)
radius_places = within["Residential Places"].fillna(0).sum()
provider_places = (
    within.groupby("Provider Name", dropna=False)["Residential Places"]
    .sum()
    .sort_values(ascending=False)
)
provider_share_pct = (
    (provider_places / radius_places) * 100 if radius_places else pd.Series(dtype=float)
)
top3_share_pct = provider_share_pct.head(3).sum() if not provider_share_pct.empty else np.nan
hhi = np.square(provider_share_pct).sum() if not provider_share_pct.empty else np.nan

summary = {
    "Facilities in radius": f"{len(within):,}",
    "Providers in radius": f"{within['Provider Name'].nunique():,}",
    "Residential places in radius": f"{int(radius_places):,}",
    "Govt funding in radius": format_money(float(within["2024-25 Australian Government Funding"].fillna(0).sum())),
    "Top-3 share (by places)": f"{top3_share_pct:,.1f}%" if pd.notna(top3_share_pct) else "N/A",
    "HHI (by places)": f"{hhi:,.0f}" if pd.notna(hhi) else "N/A",
}
summary["Funding per place in radius"] = (
    format_money(float(within["2024-25 Australian Government Funding"].fillna(0).sum() / radius_places))
    if radius_places else "N/A"
)
st.write(summary)

comp = (
    within.groupby(["Provider Name", "Organisation Type"], dropna=False)["Residential Places"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
comp["Market share %"] = np.where(
    radius_places > 0,
    (comp["Residential Places"] / radius_places) * 100,
    np.nan,
)
comp_display = comp.head(30).copy()
comp_styler = style_table(
    comp_display,
    numeric_columns=["Residential Places", "Market share %"],
    formatters={
        "Residential Places": "{:,.0f}",
        "Market share %": lambda x: f"{x:,.1f}%" if pd.notna(x) else "N/A",
    },
)
st.dataframe(comp_styler, use_container_width=True, hide_index=True)

st.caption("Next improvements: exclude the selected facility itself, add 'exclude our provider'.")
