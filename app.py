import streamlit as st
from earth import create_earth
from pathlib import Path
import base64
import pandas as pd
import threading 
from http.server import SimpleHTTPRequestHandler, HTTPServer
import streamlit.components.v1 as components
import json
import os
import plotly.express as px

create_earth()
def start_server():
    server = HTTPServer(
        ("localhost", 8000),
        SimpleHTTPRequestHandler
    )
    print("SERVER ROOT:", os.getcwd())
    server.serve_forever()


threading.Thread(
    target=start_server,
    daemon=True
).start()

df = pd.read_csv("dashboard_data/labeled_features.csv")

def load_css():

    css = Path("style.css").read_text()
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )

def set_background():

    with open("dashboard_data/background.png", "rb") as img:

        encoded = base64.b64encode(img.read()).decode()

    st.markdown(

        f"""
        <style>
        .stApp {{

            background-image:
            linear-gradient(rgba(10,12,25,0.35),
                            rgba(10,12,25,0.45)),
            url("data:image/png;base64,{encoded}");

            background-size: cover;
            background-position:center;
            background-attachment:fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Page Configuration
st.set_page_config(

    page_title="Space Debris Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"

)

load_css()
set_background()


# Dashboard Title

st.markdown("""
<h1 style="
font-family:'Bahnschrift', Arial, sans-serif;
font-size:34px;
font-weight:700;
letter-spacing:2px;
margin-bottom:0;
">
<span style="color:#F5F8FF;">
SPACE DEBRIS
</span>

<span style="color:#7EC8FF;">
COLLISION RISK
</span>

<span style="color:#F5F8FF;">
PREDICTION
</span>
</h1>
""", unsafe_allow_html=True)
st.caption(
    "Graph Attention Network based collision risk analysis for Earth orbit objects."
)

st.sidebar.image("dashboard_data/icon2.png", width=90)
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio(
    "MISSION MODULES",
    [
        "🌍 Orbital Monitor",
        "📊 System Analytic"
    ]
)

def hud_card(title, value, icon="◈"):
    components.html(f"""
        <style>
        body {{
            margin:0;
            background:transparent;
            overflow:hidden;
        }}
                
        .hud-card {{
            position:relative;
            width:180px;
            height:92px;
        }}
                
        .hud-svg {{
            position:absolute;
            pinset:0;
            width:100%;
            height:100%;
            filter:drop-shadow(0 0 4px rgba(120,180,255,.08));
        }}
                
        .info {{
            position:absolute;
            top:16px;
            left:20px;
            font-family:Segoe UI, Arial, sans-serif;
        }}
                
        .label{{
            display:flex;
            align-items:center;
            gap:8px;
                
            font-family:"Segoe UI", Arial, sans-serif;
            font-size:12px;
            font-weight:500;
            letter-spacing:2px;
            text-transform:uppercase;
                
            color:#9FB6D8 !important;
        }}
                
                
        .value {{
            margin-top:16px;
                
            font-family:"Segoe UI", Arial, sans-serif;
            color:#F5F8FF !important;
            font-size:23px;
            font-weight:400;
            line-height:1;
        }}
        </style>
                
        <div class="hud-card">
                
        <svg class="hud-svg"
            viewBox="0 0 180 92"
            preserveAspectRatio="none">
                
        <path
        d="
        M14 0
        L166 0
        L180 14
        L180 78
        L166 92
        L14 92
        L0 78
        L0 14
        Z"
        fill="rgba(8,12,35,.18)"
        stroke="#8FAFD6"
        stroke-width="0.7"/>
                
        <!-- Top Left -->
        <line x1="7" y1="16"
            x2="20" y2="3"
            stroke="#A7C3E6"
            stroke-width="0.7"/>
                
        <!-- Top Right -->
        <line x1="160" y1="3"
            x2="173" y2="16"
            stroke="#A7C3E6"
            stroke-width="0.7"/>
                
        <!-- Bottom Left -->
        <line x1="7" y1="76"
            x2="20" y2="89"
            stroke="#A7C3E6"
            stroke-width="0.7"/>
                
        <!-- Bottom Right -->
        <line x1="160" y1="89"
            x2="173" y2="76"
            stroke="#A7C3E6"
            stroke-width="0.7"/>
                
        </svg>
                
        <div class="info">
                
            <div class="label" style="color:#9FB6D8;">
                {icon}
                <span>{title}</span>
            </div>
                
            <div class="value" style="color:#F5F8FF;">
                {value}
            </div>
                
            </div>
                
            </div>
            """, height=105)

# Main Dashboard Layout
if page == "🌍 Orbital Monitor":

    left_col, right_col = st.columns([3, 2])

    # ---------- LEFT ----------
    with left_col:

        earth_path = Path("static/earth_view.html").resolve()

        st.components.v1.html(
        """
        <iframe 
            src="http://localhost:8000/static/earth_view.html"
            style="
                width:100%;
                height:700px;
                border:none;
                background:transparent;
            ">
        </iframe>
        """,
            height=600,
            scrolling=False
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2,2,6])

        with col1:
            if st.button("OBJECTS", use_container_width=True):
                Path("dashboard_data/view_mode.json").write_text(
                    '{"mode":"type"}'
                )

        with col2:
            if st.button("RISK", use_container_width=True):
                Path("dashboard_data/view_mode.json").write_text(
                    '{"mode":"risk"}'
                )
        
    # ---------- RIGHT ----------
    with right_col:

     # SEARCH

        st.markdown("""
        <h3 class='section-title'>
        SEARCH
        </h3>
        """, unsafe_allow_html=True)

        search = st.text_input(
            "",
            placeholder="⌕ Search satellite..."
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        if search:

            result = df[
                df["Satellite Name"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

            if not result.empty:

                obj = result.iloc[0]
                p = Path("dashboard_data/fly_to.json").resolve()
                print("WRITING TO:", p)
                p.write_text(
                    json.dumps({

                        "name": obj["Satellite Name"],
                        "lat": float(obj["Latitude"]),
                        "lon": float(obj["Longitude"]),
                        "alt": float(obj["Altitude (km)"]) * 1000

                    })
                )

        else:

            result = pd.DataFrame()


        # SATELLITE DETAILS
        if len(result) > 0:

            sat = result.iloc[0]
            # Find closest neighbour
            current_pos = sat[["X", "Y", "Z"]].astype(float)

            distance = (
                (df[["X","Y","Z"]] - current_pos) ** 2
            ).sum(axis=1) ** 0.5
            # remove itself
            distance[sat.name] = float("inf")
            nearest_index = distance.idxmin()
            nearest = df.loc[nearest_index]
            nearest_name = nearest["Satellite Name"]
            nearest_distance = distance[nearest_index]
            st.markdown(f"## {sat['Satellite Name']}")
            st.markdown(
                f"<span style='color:#56d8ff;'>NORAD ID: {sat['NORAD ID']}</span>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            info1, info2 = st.columns(2)

            with info1:
                st.markdown(f"""
                <div class="hud-info">
                    <div class="hud-label">CURRENT RISK</div>
                    <div class="hud-value">{sat['Risk Label']}</div>
                </div>
                """, unsafe_allow_html=True)

            with info1:

                st.markdown(f"""
                    <div style="
                        margin-top:10px;
                        font-family:'Segoe UI', Arial, sans-serif;
                        color:#F5F8FF;
                        font-size:14px;
                        letter-spacing:0.5px;
                        line-height:2;
                    ">
                    <span style="color:#FF6B7A;font-size:20px;">●</span>
                    Collision Risk:
                    <span style="color:#FF6B7A;">
                        {sat["Risk Score"]:.3f}
                    </span>

                    <br>
                    <span style="color:#FF6B7A;font-size:20px;">●</span>
                        Closest Neighbour: {nearest_name} — {nearest_distance:.2f} km
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                components.html(f"""
                <style>
                body {{
                    margin:0;
                    background:transparent;
                    overflow:hidden;
                }}

                .hud-card {{
                    position:relative;
                    width:180px;
                    height:92px;
                }}

                .hud-svg {{
                    position:absolute;
                    inset:0;
                    width:100%;
                    height:100%;
                    filter:drop-shadow(0 0 4px rgba(120,180,255,.08));
                }}

                .info {{
                    position:absolute;
                    top:16px;
                    left:20px;
                    font-family:Segoe UI, Arial, sans-serif;
                }}

                .label{{
                    display:flex;
                    align-items:center;
                    gap:8px;

                    font-family:"Segoe UI", Arial, sans-serif;
                    font-size:12px;
                    font-weight:500;
                    letter-spacing:2px;
                    text-transform:uppercase;

                    color:#9FB6D8 !important;
                }}


                .value {{
                    margin-top:16px;

                    font-family:"Segoe UI", Arial, sans-serif;
                    color:#F5F8FF !important;
                    font-size:23px;
                    font-weight:400;
                    line-height:1;
                }}
                </style>

                <div class="hud-card">

                <svg class="hud-svg"
                    viewBox="0 0 180 92"
                    preserveAspectRatio="none">

                <path
                d="
                M14 0
                L166 0
                L180 14
                L180 78
                L166 92
                L14 92
                L0 78
                L0 14
                Z"
                fill="rgba(8,12,35,.18)"
                stroke="#8FAFD6"
                stroke-width="0.7"/>

                <!-- Top Left -->
                <line x1="7" y1="16"
                    x2="20" y2="3"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>

                <!-- Top Right -->
                <line x1="160" y1="3"
                    x2="173" y2="16"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>

                <!-- Bottom Left -->
                <line x1="7" y1="76"
                    x2="20" y2="89"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>

                <!-- Bottom Right -->
                <line x1="160" y1="89"
                    x2="173" y2="76"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>

                </svg>

                <div class="info">

                    <div class="label">
                        △
                        <span>ALTITUDE</span>
                    </div>

                    <div class="value">
                        {sat["Altitude (km)"]:.1f} km
                    </div>

                 </div>

                </div>
                """, height=105)

                components.html(f"""
                <style>
                body {{
                    margin:0;
                    background:transparent;
                    overflow:hidden;
                }}
            
                .hud-card {{
                    position:relative;
                    width:180px;
                    height:92px;
                }}
            
                .hud-svg {{
                    position:absolute;
                    inset:0;
                    width:100%;
                    height:100%;
                    filter:drop-shadow(0 0 4px rgba(120,180,255,.08));
                }}
            
                .info {{
                    position:absolute;
                    top:16px;
                    left:20px;
                    font-family:Segoe UI, Arial, sans-serif;
                }}
            
                .label{{
                    display:flex;
                    align-items:center;
                    gap:8px;
            
                    font-family:"Segoe UI", Arial, sans-serif;
                    font-size:12px;
                    font-weight:500;
                    letter-spacing:2px;
                    text-transform:uppercase;
            
                    color:#9FB6D8 !important;
                }}
            
            
                .value {{
                    margin-top:16px;
            
                    font-family:"Segoe UI", Arial, sans-serif;
                    color:#F5F8FF !important;
                    font-size:23px;
                    font-weight:400;
                    line-height:1;
                }}
                </style>
            
                <div class="hud-card">
            
                <svg class="hud-svg"
                    viewBox="0 0 180 92"
                    preserveAspectRatio="none">
            
                <path
                d="
                M14 0
                L166 0
                L180 14
                L180 78
                L166 92
                L14 92
                L0 78
                L0 14
                Z"
                fill="rgba(8,12,35,.18)"
                stroke="#8FAFD6"
                stroke-width="0.7"/>
            
                <!-- Top Left -->
                <line x1="7" y1="16"
                    x2="20" y2="3"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Top Right -->
                <line x1="160" y1="3"
                    x2="173" y2="16"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Bottom Left -->
                <line x1="7" y1="76"
                    x2="20" y2="89"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Bottom Right -->
                <line x1="160" y1="89"
                    x2="173" y2="76"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                </svg>
            
                <div class="info">
            
                    <div class="label" style="color:#9FB6D8;">
                        ↗
                        <span>VELOCITY</span>
                    </div>
            
                    <div class="value" style="color:#F5F8FF;">
                        {sat["Speed (km/s)"]:.2f} km/s
                    </div>
            
                </div>
            
                </div>
                """, height=105)

            with col2:
                components.html(f"""
                <style>
                 body {{
                    margin:0;
                    background:transparent;
                    overflow:hidden;
                }}
            
                .hud-card {{
                    position:relative;
                    width:180px;
                    height:92px;
                }}
            
                .hud-svg {{
                    position:absolute;
                    inset:0;
                    width:100%;
                    height:100%;
                    filter:drop-shadow(0 0 4px rgba(120,180,255,.08));
                }}
            
                .info {{
                    position:absolute;
                    top:16px;
                    left:20px;
                    font-family:Segoe UI, Arial, sans-serif;
                }}
            
                .label{{
                    display:flex;
                    align-items:center;
                    gap:8px;
            
                    font-family:"Segoe UI", Arial, sans-serif;
                    font-size:12px;
                    font-weight:500;
                    letter-spacing:2px;
                    text-transform:uppercase;
            
                    color:#9FB6D8 !important;
                }}
            
            
                .value {{
                    margin-top:16px;
            
                    font-family:"Segoe UI", Arial, sans-serif;
                    color:#F5F8FF !important;
                    font-size:23px;
                    font-weight:400;
                    line-height:1;
                }}
                </style>
            
                <div class="hud-card">
            
                <svg class="hud-svg"
                    viewBox="0 0 180 92"
                    preserveAspectRatio="none">
            
                <path
                d="
                M14 0
                L166 0
                L180 14
                L180 78
                L166 92
                L14 92
                L0 78
                L0 14
                Z"
                fill="rgba(8,12,35,.18)"
                stroke="#8FAFD6"
                stroke-width="0.7"/>
            
                <!-- Top Left -->
                <line x1="7" y1="16"
                    x2="20" y2="3"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Top Right -->
                <line x1="160" y1="3"
                    x2="173" y2="16"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Bottom Left -->
                <line x1="7" y1="76"
                    x2="20" y2="89"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Bottom Right -->
                <line x1="160" y1="89"
                    x2="173" y2="76"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                </svg>
            
                <div class="info">
            
                    <div class="label" style="color:#9FB6D8;">
                        ◈
                        <span>OBJECT TYPE</span>
                    </div>
            
                    <div class="value" style="color:#F5F8FF;">
                        {sat["Object Type"]}
                    </div>
            
                </div>
            
                </div>
                """, height=105)

                components.html(f"""
                <style>
                body {{
                    margin:0;
                    background:transparent;
                    overflow:hidden;
                }}
            
                .hud-card {{
                    position:relative;
                    width:180px;
                    height:92px;
                }}
            
                .hud-svg {{
                    position:absolute;
                    inset:0;
                    width:100%;
                    height:100%;
                    filter:drop-shadow(0 0 4px rgba(120,180,255,.08));
                }}
            
                .info {{
                    position:absolute;
                    top:16px;
                    left:20px;
                    font-family:Segoe UI, Arial, sans-serif;
                }}
            
                .label{{
                    display:flex;
                    align-items:center;
                    gap:8px;
            
                    font-family:"Segoe UI", Arial, sans-serif;
                    font-size:12px;
                    font-weight:500;
                    letter-spacing:2px;
                    text-transform:uppercase;
            
                    color:#9FB6D8 !important;
                }}
            
            
                .value {{
                    margin-top:16px;
            
                    font-family:"Segoe UI", Arial, sans-serif;
                    color:#F5F8FF !important;
                    font-size:23px;
                    font-weight:400;
                    line-height:1;
                }}
                </style>
            
                <div class="hud-card">
            
                <svg class="hud-svg"
                    viewBox="0 0 180 92"
                    preserveAspectRatio="none">
            
                <path
                d="
                M14 0
                L166 0
                L180 14
                L180 78
                L166 92
                L14 92
                L0 78
                L0 14
                Z"
                fill="rgba(8,12,35,.18)"
                stroke="#8FAFD6"
                stroke-width="0.7"/>
            
                <!-- Top Left -->
                <line x1="7" y1="16"
                    x2="20" y2="3"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Top Right -->
                <line x1="160" y1="3"
                    x2="173" y2="16"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Bottom Left -->
                <line x1="7" y1="76"
                    x2="20" y2="89"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                <!-- Bottom Right -->
                <line x1="160" y1="89"
                    x2="173" y2="76"
                    stroke="#A7C3E6"
                    stroke-width="0.7"/>
            
                </svg>
            
                <div class="info">
            
                    <div class="label" style="color:#9FB6D8;">
                        ⤴
                        <span>INCLINATION</span>
                    </div>
            
                    <div class="value" style="color:#F5F8FF;">
                        {sat["Inclination"]:.2f}°
                    </div>
            
                </div>
            
                </div>
                """, height=105)

        else:

            st.caption("Search for a satellite to begin analysis.")

        

 # ANALYTICS
elif page == "📊 System Analytic":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <h3 class='section-title'>
    SYSTEM ANALYTICS
    </h3>
    """, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
        color:#9FB6D8;
        letter-spacing:1.5px;
        font-size:13px;">
        GLOBAL ORBITAL ENVIRONMENT OVERVIEW
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    sat_count = (df["Object Type"] == "Satellite").sum()
    debris_count = (df["Object Type"] == "Debris").sum()
    risk_count = (df["Risk Label"] == "High Risk").sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        hud_card("TOTAL OBJECTS", f"{len(df):,}", "◎")

    with c2:
        hud_card("SATELLITES", f"{sat_count:,}", "🛰")

    with c3:
        hud_card("DEBRIS", f"{debris_count:,}", "☄")

    with c4:
        hud_card("ELEVATED RISK", f"{risk_count:,}", "⚠")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Object Composition")

        type_counts = df["Object Type"].value_counts()

        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            hole=0.55
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with col2:
        st.markdown("### Risk Distribution")

        risk_counts = df["Risk Label"].replace({
            "High Risk": "Elevated Risk",
            "Low Risk": "Nominal Risk"
        }).value_counts()

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            hole=0.55
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(
            fig, 
            use_container_width=True, 
            config={"displayModeBar": False}
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Altitude Distribution")

    altitude_data = df[
        df["Altitude (km)"] < 5000
    ]

    fig = px.histogram(
        df,
        x="Altitude (km)",
        nbins=40
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Altitude (km)",
        yaxis_title="Objects"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )


