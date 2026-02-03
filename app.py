# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

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
SUMMARY_MAP_HEIGHT = 520
df = load_data(DATA_PATH)

logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.image("malogo1.svg", width=140)
with title_col:
    st.title("Residential Aged Care Provider Dashboard")
    st.caption(
        "Source: [AIHW Aged Care Service list, 30 June 2025]"
        "(https://www.gen-agedcaredata.gov.au/resources/access-data/2025/october/aged-care-service-list-30-june-2025)"
    )

# Sidebar filters
st.sidebar.header("Filters")

state = st.sidebar.multiselect(
    "State",
    options=sorted(df["Physical State"].dropna().unique())
)

org_type = st.sidebar.multiselect(
    "Organisation Type",
    options=sorted(df["Organisation Type"].dropna().unique())
)

residential_df = df.copy()
if "Care Type" in residential_df.columns:
    residential_df = residential_df[
        residential_df["Care Type"].astype(str).str.strip().str.lower() == "residential"
    ]

acpr_source = residential_df.copy()
if state:
    acpr_source = acpr_source[acpr_source["Physical State"].isin(state)]

acpr = st.sidebar.multiselect(
    "Aged Care Planning Region",
    options=sorted(acpr_source["2018 Aged Care Planning Region (ACPR)"].dropna().unique())
)

provider_source = residential_df.copy()
if state:
    provider_source = provider_source[provider_source["Physical State"].isin(state)]
if org_type:
    provider_source = provider_source[provider_source["Organisation Type"].isin(org_type)]
if acpr:
    provider_source = provider_source[
        provider_source["2018 Aged Care Planning Region (ACPR)"].isin(acpr)
    ]
provider_filter = st.sidebar.multiselect(
    "Provider",
    options=sorted(provider_source["Provider Name"].dropna().unique()),
)

dff = residential_df.copy()
if state:
    dff = dff[dff["Physical State"].isin(state)]
if org_type:
    dff = dff[dff["Organisation Type"].isin(org_type)]
if acpr:
    dff = dff[dff["2018 Aged Care Planning Region (ACPR)"].isin(acpr)]
if provider_filter:
    dff = dff[dff["Provider Name"].isin(provider_filter)]

def apply_sidebar_filters(base_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    if state:
        out = out[out["Physical State"].isin(state)]
    if org_type:
        out = out[out["Organisation Type"].isin(org_type)]
    if acpr:
        out = out[out["2018 Aged Care Planning Region (ACPR)"].isin(acpr)]
    if provider_filter:
        out = out[out["Provider Name"].isin(provider_filter)]
    return out

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: #fbf4f8;
        border: 1px solid #f0d4e6;
        border-left: 4px solid #D6008F;
        border-radius: 10px;
        padding: 8px 10px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        font-size: 0.95rem !important;
        line-height: 1.35;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.95rem !important;
        line-height: 1.35 !important;
    }
    .summary-section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 2px solid #d7d7d7;
        padding-left: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ece7db;
        border: 1px solid #c8bea8;
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #5c4f35;
        padding: 0.45rem 1rem 0.4rem 1rem;
        margin-top: 0.35rem;
    }
    .stTabs [aria-selected="true"] {
        background: #fffdf8;
        color: #2a2520;
        border-color: #b6ab92;
        margin-top: 0;
        transform: translateY(2px);
    }
    .comp-instruction {
        font-size: 1.02rem;
        font-weight: 600;
        color: #3e3e3e;
        margin: 0.2rem 0 0.55rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

view_mode = st.segmented_control(
    "View",
    options=["Overview", "Competitor analysis"],
    default="Overview",
    key="active_view",
    label_visibility="collapsed",
    width="stretch",
)

if view_mode == "Overview":
    if dff.empty:
        st.warning("No rows match the current sidebar filters. Adjust filters to see overview outputs.")
    else:
        top_block = st.container(border=True)
        with top_block:
            top_left, top_right = st.columns([1.15, 0.85], gap="large")

            with top_left:
                st.subheader("Summary statistics")
                c1, c2, c3 = st.columns(3, gap="small")
                c1.metric("Facilities", f"{len(dff):,}")
                c2.metric("Providers", f"{dff['Provider Name'].nunique():,}")
                c3.metric("Residential places", f"{int(dff['Residential Places'].fillna(0).sum()):,}")

                funding_sum = dff["2024-25 Australian Government Funding"].fillna(0).sum()
                places_sum = dff["Residential Places"].fillna(0).sum()
                provider_count = dff["Provider Name"].nunique()
                avg_funding_per_facility = funding_sum / len(dff) if len(dff) else np.nan
                avg_funding_per_provider = funding_sum / provider_count if provider_count else np.nan
                avg_funding_per_place = funding_sum / places_sum if places_sum else np.nan

                c4 = st.columns(1)[0]
                c4.metric("Total government funding", format_money(funding_sum))

                c5, c6, c7 = st.columns(3, gap="small")
                c5.metric("Avg funding / facility", format_money(avg_funding_per_facility))
                c6.metric("Avg funding / provider", format_money(avg_funding_per_provider))
                c7.metric("Avg funding / place", format_money(avg_funding_per_place))

            with top_right:
                st.subheader("Map")
                map_df = dff.dropna(subset=["Latitude", "Longitude"]).copy()
                if map_df.empty:
                    st.info("No map points for the current filters.")
                else:
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position="[Longitude, Latitude]",
                        get_radius=1200,
                        radius_min_pixels=5,
                        radius_max_pixels=18,
                        get_fill_color=[214, 0, 143, 230],
                        get_line_color=[255, 255, 255, 255],
                        line_width_min_pixels=1,
                        stroked=True,
                        pickable=True,
                    )
                    view_state = pdk.ViewState(
                        latitude=-26.5,
                        longitude=134.5,
                        zoom=2.7,
                        pitch=0,
                        bearing=0,
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            map_style="light",
                            tooltip={"text": "{Service Name}\n{Provider Name}\n{Physical Suburb}, {Physical State}"},
                        ),
                        use_container_width=True,
                        height=SUMMARY_MAP_HEIGHT,
                    )

        st.divider()
        provider_table = (
            dff.groupby(["Provider Name", "Organisation Type"], dropna=False)[
                ["Residential Places", "2024-25 Australian Government Funding"]
            ]
            .sum()
            .reset_index()
        )
        provider_table["Funding per place"] = np.where(
            provider_table["Residential Places"] > 0,
            provider_table["2024-25 Australian Government Funding"] / provider_table["Residential Places"],
            np.nan,
        )
        provider_table = provider_table.sort_values("Residential Places", ascending=False).head(30)
        providers_block = st.container(border=True)
        with providers_block:
            st.subheader(f"Largest {len(provider_table):,} providers")
            provider_styler = style_table(
                provider_table,
                numeric_columns=["Residential Places", "2024-25 Australian Government Funding", "Funding per place"],
                formatters={
                    "Residential Places": "{:,.0f}",
                    "2024-25 Australian Government Funding": lambda x: f"${np.round(x / 1000) * 1000:,.0f}" if pd.notna(x) else "N/A",
                    "Funding per place": lambda x: f"${np.round(x / 1000) * 1000:,.0f}" if pd.notna(x) else "N/A",
                },
            )
            st.dataframe(provider_styler, use_container_width=True, hide_index=True)

        st.divider()
        funding_block = st.container(border=True)
        with funding_block:
            st.subheader("Funding breakdown")
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
                    .mark_bar(color="#D6008F")
                    .encode(
                        x=alt.X(
                            "Physical State:N",
                            sort="-y",
                            axis=alt.Axis(labelAngle=-45, labelLimit=120, title=None),
                        ),
                        y=alt.Y("Funding_m:Q", axis=alt.Axis(format=",.1f", title="Funding ($m)")),
                        tooltip=[
                            alt.Tooltip("Physical State:N", title="State"),
                            alt.Tooltip("Funding:Q", format="$,.0f"),
                        ],
                    )
                    .properties(height=320)
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
                    .mark_bar(color="#D6008F")
                    .encode(
                        x=alt.X(
                            "Organisation Type:N",
                            sort="-y",
                            axis=alt.Axis(labelAngle=-45, labelLimit=120, title=None),
                        ),
                        y=alt.Y("Funding_m:Q", axis=alt.Axis(format=",.1f", title="Funding ($m)")),
                        tooltip=[
                            alt.Tooltip("Organisation Type:N"),
                            alt.Tooltip("Funding:Q", format="$,.0f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(org_chart, use_container_width=True)

if view_mode == "Competitor analysis":
    st.info("How to use this tab: Select a facility, then set a radius to analyse nearby competitors.")
    use_sidebar_filters = st.toggle("Apply sidebar filters in this tab", value=False)
    comp_df = apply_sidebar_filters(residential_df) if use_sidebar_filters else residential_df.copy()

    facility_options = (
        comp_df[["Service Name", "Provider Name", "Physical Suburb", "Physical State", "Latitude", "Longitude"]]
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
        st.info("No facilities with coordinates are available for this competition scope.")
    else:
        selected = st.selectbox("Select a facility", options=facility_options["label"].tolist())
        radius_km = st.slider("Radius (km)", min_value=1, max_value=80, value=10)

        sel_row = facility_options[facility_options["label"] == selected].iloc[0]
        lat0, lon0 = float(sel_row["Latitude"]), float(sel_row["Longitude"])

        cand = comp_df.dropna(subset=["Latitude", "Longitude"]).copy()
        cand["dist_km"] = haversine_km(lat0, lon0, cand["Latitude"].astype(float), cand["Longitude"].astype(float))
        within = cand[cand["dist_km"] <= radius_km].copy()

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

        funding_in_radius = float(within["2024-25 Australian Government Funding"].fillna(0).sum())
        funding_per_place_in_radius = (funding_in_radius / radius_places) if radius_places else np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Facilities in radius", f"{len(within):,}")
        m2.metric("Providers in radius", f"{within['Provider Name'].nunique():,}")
        m3.metric("Residential places in radius", f"{int(radius_places):,}")
        m4.metric("Govt funding in radius", format_money(funding_in_radius))

        m5, m6, m7 = st.columns(3)
        m5.metric(
            "Funding per place in radius",
            format_money(funding_per_place_in_radius) if pd.notna(funding_per_place_in_radius) else "N/A",
        )
        m6.metric(
            "Top-3 share (by places)",
            f"{top3_share_pct:,.1f}%" if pd.notna(top3_share_pct) else "N/A",
            help="Provides the proportion of places provided by the three largest providers in the given radius.",
        )
        m7.metric(
            "HHI (by places)",
            f"{hhi:,.0f}" if pd.notna(hhi) else "N/A",
            help=(
                "Provides a measure of market concentration in the given radius.\n"
                "\n"
                "> <1,000: Competitive marketplace (low concentration).\n"
                ">\n"
                "> 1,500-2,000: Moderate concentration.\n"
                ">\n"
                "> \\> 2,000: High concentration (monopoly potential)."
            ),
        )

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
