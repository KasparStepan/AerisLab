"""
AerisLab Postprocessing Dashboard

Interactive Streamlit application for visualizing and exporting simulation results.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import tempfile
import io

# -- Import plotting module --
try:
    import aerislab.visualization.plotting as plotting
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import aerislab.visualization.plotting as plotting

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="AerisLab Postprocessing",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 AerisLab Postprocessing Dashboard")

# =============================================================================
# Sidebar - Data Selection
# =============================================================================

st.sidebar.header("📁 Data Selection")

# 1. Automatic File Discovery (sorted by date, newest first)
output_dir = Path("output")
discovered_files = []
if output_dir.exists():
    discovered_files = list(output_dir.rglob("simulation.csv"))
    discovered_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

# Build options dict with readable labels
file_options = {}
for f in discovered_files:
    # Extract simulation name from path structure: output/<sim_name>/logs/simulation.csv
    sim_name = f.parent.parent.name
    file_options[sim_name] = f

# File upload or selection
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload a simulation CSV file")
selected_file_path = None

if uploaded_file:
    selected_file_path = uploaded_file
elif file_options:
    selection = st.sidebar.selectbox(
        "Or select existing simulation",
        options=list(file_options.keys()),
        help="Simulations found in 'output/' directory, sorted by date"
    )
    selected_file_path = file_options[selection]
else:
    st.info("👆 No simulations found in 'output/'. Please upload a CSV file.")
    st.stop()

# =============================================================================
# Load Data
# =============================================================================

@st.cache_data
def load_data(file):
    """Load CSV data with caching."""
    if hasattr(file, 'read'):
        return pd.read_csv(file)
    return pd.read_csv(file)

try:
    df = load_data(selected_file_path)
    st.sidebar.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    st.stop()

# =============================================================================
# Sidebar - Body & Plot Selection
# =============================================================================

st.sidebar.header("📊 Plot Configuration")

# Auto-detect bodies from column names
bodies = set()
for col in df.columns:
    if ".p_x" in col:
        bodies.add(col.split(".p_x")[0])

body_list = sorted(list(bodies)) if bodies else ["payload"]
body_name = st.sidebar.selectbox("Select Body", options=body_list)

plot_type = st.sidebar.radio(
    "Plot Type",
    ["Trajectory (3D)", "Kinematics", "Forces"],
    help="Choose the type of visualization"
)

# =============================================================================
# Sidebar - Figure Options
# =============================================================================

st.sidebar.header("🎨 Figure Styling")

with st.sidebar.expander("📐 Dimensions", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        width_px = st.number_input("Width (px)", value=1000, step=50, min_value=400)
    with col2:
        height_px = st.number_input("Height (px)", value=700, step=50, min_value=300)
    figsize = (width_px / 100.0, height_px / 100.0)

with st.sidebar.expander("🔤 Typography", expanded=True):
    font_size = st.slider("Font Size", min_value=8, max_value=24, value=14)
    custom_title = st.text_input("Plot Title", value="", placeholder="Auto-generated")

with st.sidebar.expander("✏️ Labels", expanded=False):
    if plot_type == "Trajectory (3D)":
        custom_xlabel = st.text_input("X Label", value="X [m]")
        custom_ylabel = st.text_input("Y Label", value="Y [m]")
        custom_zlabel = st.text_input("Z Label", value="Z [m]")
    else:
        custom_xlabel = st.text_input("X Label", value="Time [s]")
        custom_ylabel = None
        custom_zlabel = None

with st.sidebar.expander("🎨 Appearance", expanded=True):
    line_color = st.color_picker("Line Color", value="#1f77b4")
    line_width = st.slider("Line Width", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    marker_size = st.slider("Marker Size", min_value=4, max_value=16, value=8)
    show_grid = st.checkbox("Show Grid", value=True)
    show_legend = st.checkbox("Show Legend", value=True)
    
    if plot_type in ["Kinematics", "Forces"]:
        show_magnitude = st.checkbox("Show Magnitude", value=True, help="Plot vector magnitude vs components")
    else:
        show_magnitude = True

# =============================================================================
# Main Area - Data Preview
# =============================================================================

with st.expander("📋 Data Preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Showing first 20 of {len(df):,} rows")

# =============================================================================
# Main Area - Plot Generation
# =============================================================================

st.divider()

# Create temp file for plotting (using proper tempfile)
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
    df.to_csv(tmp.name, index=False)
    temp_csv_path = tmp.name

try:
    # Clean title (empty string -> None for auto-generation)
    title = custom_title.strip() if custom_title.strip() else None
    
    if plot_type == "Trajectory (3D)":
        st.subheader(f"🛸 3D Trajectory: {body_name}")
        
        fig = plotting.plot_trajectory_3d(
            temp_csv_path,
            body_name,
            engine="plotly",
            show=False,
            figsize=figsize,
            title=title,
            xlabel=custom_xlabel,
            ylabel=custom_ylabel,
            zlabel=custom_zlabel,
            font_size=font_size,
            line_color=line_color,
            line_width=line_width,
            marker_size=marker_size,
            show_grid=show_grid,
            show_legend=show_legend,
        )
        
    elif plot_type == "Kinematics":
        st.subheader(f"📈 Kinematics: {body_name}")
        
        fig = plotting.plot_velocity_and_acceleration(
            temp_csv_path,
            body_name,
            engine="plotly",
            show=False,
            magnitude=show_magnitude,
            figsize=figsize,
            title=title,
            xlabel=custom_xlabel,
            font_size=font_size,
            line_width=line_width,
            show_grid=show_grid,
            show_legend=show_legend,
        )
        
    elif plot_type == "Forces":
        st.subheader(f"⚡ Forces & Torques: {body_name}")
        
        fig = plotting.plot_forces(
            temp_csv_path,
            body_name,
            engine="plotly",
            show=False,
            magnitude=show_magnitude,
            figsize=figsize,
            title=title,
            xlabel=custom_xlabel,
            font_size=font_size,
            line_width=line_width,
            show_grid=show_grid,
            show_legend=show_legend,
        )

    # Display the figure
    if fig:
        st.plotly_chart(fig, use_container_width=False)
        
        # =============================================================================
        # Export Options
        # =============================================================================
        
        st.divider()
        st.subheader("💾 Export Figure")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PNG Export
            img_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name=f"{body_name}_{plot_type.lower().replace(' ', '_')}.png",
                mime="image/png",
            )
        
        with col2:
            # SVG Export
            svg_bytes = fig.to_image(format="svg")
            st.download_button(
                label="📥 Download SVG",
                data=svg_bytes,
                file_name=f"{body_name}_{plot_type.lower().replace(' ', '_')}.svg",
                mime="image/svg+xml",
            )
        
        with col3:
            # HTML Export (interactive)
            html_str = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📥 Download HTML",
                data=html_str,
                file_name=f"{body_name}_{plot_type.lower().replace(' ', '_')}.html",
                mime="text/html",
            )
    else:
        st.warning("No figure generated. Check that the body name is correct.")

except Exception as e:
    st.error(f"❌ Error generating plot: {e}")
    with st.expander("🔍 Debug Info"):
        st.write("**Available columns:**")
        st.write(list(df.columns))
        st.write(f"**Body name:** {body_name}")
        st.write(f"**Expected columns:** {body_name}.p_x, {body_name}.p_y, {body_name}.p_z")

finally:
    # Cleanup temp file
    try:
        Path(temp_csv_path).unlink()
    except:
        pass

# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption("AerisLab Postprocessing Dashboard | Built with Streamlit & Plotly")
