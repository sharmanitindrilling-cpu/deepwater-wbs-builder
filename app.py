
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Deepwater WBS Builder", layout="wide")

# -----------------------------
# Core engineering calculations
# -----------------------------
def minimum_curvature_tvd(df):
    """Calculate TVD, Northing, Easting and DLS from MD/Inc/Azi survey."""
    d = df.sort_values("MD_ft").reset_index(drop=True).copy()
    n = len(d)
    tvd = np.zeros(n)
    north = np.zeros(n)
    east = np.zeros(n)
    dls = np.zeros(n)

    # Assumption: first station TVD equals first MD unless supplied.
    tvd[0] = d.loc[0, "TVD_ft_optional"] if "TVD_ft_optional" in d and pd.notna(d.loc[0, "TVD_ft_optional"]) else d.loc[0, "MD_ft"]

    for i in range(1, n):
        md1, md2 = float(d.loc[i-1, "MD_ft"]), float(d.loc[i, "MD_ft"])
        inc1 = math.radians(float(d.loc[i-1, "Inclination_deg"]))
        inc2 = math.radians(float(d.loc[i, "Inclination_deg"]))
        azi1 = math.radians(float(d.loc[i-1, "Azimuth_deg"]))
        azi2 = math.radians(float(d.loc[i, "Azimuth_deg"]))
        dmd = md2 - md1

        cos_dogleg = (
            math.cos(inc1) * math.cos(inc2)
            + math.sin(inc1) * math.sin(inc2) * math.cos(azi2 - azi1)
        )
        cos_dogleg = max(-1.0, min(1.0, cos_dogleg))
        dogleg = math.acos(cos_dogleg)
        rf = 1.0 if abs(dogleg) < 1e-12 else (2.0 / dogleg) * math.tan(dogleg / 2.0)

        dtvd = 0.5 * dmd * (math.cos(inc1) + math.cos(inc2)) * rf
        dn = 0.5 * dmd * (math.sin(inc1)*math.cos(azi1) + math.sin(inc2)*math.cos(azi2)) * rf
        de = 0.5 * dmd * (math.sin(inc1)*math.sin(azi1) + math.sin(inc2)*math.sin(azi2)) * rf

        tvd[i] = tvd[i-1] + dtvd
        north[i] = north[i-1] + dn
        east[i] = east[i-1] + de
        dls[i] = math.degrees(dogleg) * 100.0 / dmd if dmd else 0.0

    d["TVD_calc_ft"] = tvd
    d["Northing_calc_ft"] = north
    d["Easting_calc_ft"] = east
    d["DLS_calc_deg_per_100ft"] = dls

    # Prefer supplied TVD if complete; otherwise use minimum-curvature TVD.
    if "TVD_ft_optional" in d.columns and d["TVD_ft_optional"].notna().all():
        d["TVD_ft"] = d["TVD_ft_optional"].astype(float)
    else:
        d["TVD_ft"] = d["TVD_calc_ft"]
    return d


def interp_by_md(survey, md, column):
    x = survey["MD_ft"].to_numpy(float)
    y = survey[column].to_numpy(float)
    if md < x.min() or md > x.max():
        return np.nan
    return float(np.interp(md, x, y))


def md_solutions_from_tvd(survey, target_tvd):
    """Return all approximate MD solutions where the survey crosses target TVD."""
    md = survey["MD_ft"].to_numpy(float)
    tvd = survey["TVD_ft"].to_numpy(float)
    sols = []
    for i in range(len(md)-1):
        a, b = tvd[i], tvd[i+1]
        if target_tvd == a:
            sols.append(float(md[i]))
        if (target_tvd-a)*(target_tvd-b) <= 0 and a != b:
            f = (target_tvd-a)/(b-a)
            if 0 <= f <= 1:
                sols.append(float(md[i] + f*(md[i+1]-md[i])))
    # de-duplicate
    out = []
    for x in sols:
        if not out or abs(x-out[-1]) > 0.1:
            out.append(x)
    return out


def temp_at_tvd(temp_df, tvd):
    t = temp_df.dropna(subset=["TVD_ft","Temperature_F"]).sort_values("TVD_ft")
    if t.empty or tvd < t["TVD_ft"].min() or tvd > t["TVD_ft"].max():
        return np.nan
    return float(np.interp(tvd, t["TVD_ft"], t["Temperature_F"]))



def calculate_shoe_data(casing, survey, temp, well_info, fitlot):
    rkb = float(well_info.get("RKB", 0) or 0)
    out = []
    for _, row in casing.dropna(subset=["String_ID","Shoe_MD_ft"]).iterrows():
        md = float(row["Shoe_MD_ft"])
        tvd = interp_by_md(survey, md, "TVD_ft")
        inc = interp_by_md(survey, md, "Inclination_deg")
        azi = interp_by_md(survey, md, "Azimuth_deg")
        subset = survey[survey["MD_ft"] <= md]
        max_inc = float(subset["Inclination_deg"].max()) if not subset.empty else np.nan
        temperature = temp_at_tvd(temp, tvd) if pd.notna(tvd) else np.nan

        top_md = float(row["Top_MD_ft"]) if pd.notna(row.get("Top_MD_ft")) else np.nan
        top_tvd = interp_by_md(survey, top_md, "TVD_ft") if pd.notna(top_md) else np.nan
        toc_tvd = interp_by_md(survey, float(row["TOC_MD_ft"]), "TVD_ft") if pd.notna(row.get("TOC_MD_ft")) else np.nan

        tests = fitlot[fitlot["String_ID"] == row["String_ID"]] if not fitlot.empty else pd.DataFrame()
        test_type = tests.iloc[-1]["Test_Type"] if not tests.empty else ""
        test_val = tests.iloc[-1]["Result_EMW_ppg"] if not tests.empty else np.nan

        out.append({
            "String_ID": row["String_ID"],
            "String_Name": row["String_Name"],
            "Hole_Size_in": row.get("Hole_Size_in", np.nan),
            "Casing_OD_in": row.get("Casing_OD_in", np.nan),
            "Casing_ID_in": row.get("Casing_ID_in", np.nan),
            "Drift_in": row.get("Drift_in", np.nan),
            "Casing_Grade": row.get("Casing_Grade", ""),
            "Casing_Top_MD_ft": top_md,
            "Casing_Top_TVD_ft": top_tvd,
            "Shoe_MD_ft": md,
            "Shoe_TVD_ft": tvd,
            "Shoe_TVDSS_ft": tvd - rkb if pd.notna(tvd) else np.nan,
            "Inc_at_Shoe_deg": inc,
            "Max_Inc_to_Shoe_deg": max_inc,
            "Azimuth_at_Shoe_deg": azi,
            "Temperature_F": temperature,
            "FIT_LOT_Type": test_type,
            "FIT_LOT_Value": test_val,
            "TOC_TVD_ft": toc_tvd,
            "Mud_Weight_Min_ppg": row.get("Mud_Weight_Min_ppg", np.nan),
            "Mud_Weight_Max_ppg": row.get("Mud_Weight_Max_ppg", np.nan),
            "Mud_Type": row.get("Mud_Type", ""),
        })
    return pd.DataFrame(out)


def calculate_geo_data(geo, survey, temp, rkb):
    out = []
    for _, row in geo.dropna(subset=["Top_ID","Formation_Name","Input_Depth_ft"]).iterrows():
        typ = str(row["Input_Depth_Type"]).strip().upper()
        depth = float(row["Input_Depth_ft"])
        if typ == "MD":
            md = depth
            tvd = interp_by_md(survey, md, "TVD_ft")
        else:
            sols = md_solutions_from_tvd(survey, depth)
            md = sols[0] if len(sols) == 1 else (sols[0] if sols else np.nan)
            tvd = depth
        inc = interp_by_md(survey, md, "Inclination_deg") if pd.notna(md) else np.nan
        azi = interp_by_md(survey, md, "Azimuth_deg") if pd.notna(md) else np.nan
        out.append({
            "Top_ID": row["Top_ID"],
            "Formation_Name": row["Formation_Name"],
            "MD_ft": md,
            "TVD_ft": tvd,
            "TVDSS_ft": tvd - rkb if pd.notna(tvd) else np.nan,
            "Inclination_deg": inc,
            "Azimuth_deg": azi,
            "Temperature_F": temp_at_tvd(temp, tvd) if pd.notna(tvd) else np.nan,
            "Category": row.get("Category",""),
        })
    return pd.DataFrame(out)


# -----------------------------
# Workbook import/export helpers
# -----------------------------
def read_workbook(file):
    xls = pd.ExcelFile(file)
    sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}

    wi = sheets["Well_Info"].iloc[:, :4].copy()
    wi.columns = ["Field","Value","Unit","Notes"]
    wi = wi.dropna(subset=["Field"])
    well = dict(zip(wi["Field"], wi["Value"]))

    casing = sheets["Casing_Program"].copy()
    survey = sheets["Directional_Survey"].copy()
    geo = sheets["Geological_Tops"].copy()
    temp = sheets["Temperature_Gradient"].copy()
    fitlot = sheets["FIT_LOT"].copy()

    # Template tables begin at row 3; remove empty/title rows if pandas captured them.
    def normalize(df, expected_header):
        if expected_header in df.columns:
            return df
        raw = pd.read_excel(file, sheet_name=df.attrs.get("sheet_name",""), header=None)
        return df

    # The template's third Excel row is the header, so re-read with header=2.
    casing = pd.read_excel(file, sheet_name="Casing_Program", header=2)
    survey = pd.read_excel(file, sheet_name="Directional_Survey", header=2)
    geo = pd.read_excel(file, sheet_name="Geological_Tops", header=2)
    temp = pd.read_excel(file, sheet_name="Temperature_Gradient", header=2)
    fitlot = pd.read_excel(file, sheet_name="FIT_LOT", header=2)

    return well, casing, survey, geo, temp, fitlot


def make_output_excel(well, casing, survey, geo, temp, fitlot, shoes, geocalc):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame({"Field": list(well.keys()), "Value": list(well.values())}).to_excel(writer, "Well_Info", index=False)
        casing.to_excel(writer, "Casing_Program", index=False)
        survey.to_excel(writer, "Directional_Survey_Calc", index=False)
        geo.to_excel(writer, "Geological_Tops", index=False)
        temp.to_excel(writer, "Temperature_Gradient", index=False)
        fitlot.to_excel(writer, "FIT_LOT", index=False)
        shoes.to_excel(writer, "Calculated_Shoe_Data", index=False)
        geocalc.to_excel(writer, "Calculated_Geo_Data", index=False)
    bio.seek(0)
    return bio.getvalue()


# -----------------------------
# Schematic plotting
# -----------------------------


def draw_wbs(well, casing, shoes, geocalc, fitlot=None):
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12.5, 16))

    all_depths = []
    if not casing.empty and "Shoe_MD_ft" in casing:
        all_depths += casing["Shoe_MD_ft"].dropna().astype(float).tolist()
    if not geocalc.empty and "MD_ft" in geocalc:
        all_depths += geocalc["MD_ft"].dropna().astype(float).tolist()
    max_depth = max(all_depths) if all_depths else 10000.0

    rkb = float(well.get("RKB", 0) or 0)
    water_depth = float(well.get("Water_Depth", 0) or 0)
    mudline_md = float(well.get("Mudline_TVD_RKB", water_depth + rkb) or (water_depth + rkb))
    td_md = float(well.get("Planned_TD_MD", max_depth) or max_depth)
    max_depth = max(max_depth, td_md)

    geo_left, geo_right = -6.0, -3.1
    well_center = 0.0
    section_x = 2.0
    callout_x = 4.8
    right_edge = 9.0

    well_name = str(well.get("Well_Name", "Deepwater Well"))
    prospect = str(well.get("Field_Prospect", ""))

    ax.text((geo_left + right_edge)/2, -0.06*max_depth, "WELLBORE DIAGRAM",
            ha="center", fontsize=17, fontweight="bold")
    ax.text((geo_left + right_edge)/2, -0.027*max_depth, well_name,
            ha="center", fontsize=13, fontweight="bold")
    if prospect and prospect != "nan":
        ax.text((geo_left + right_edge)/2, -0.006*max_depth, prospect,
                ha="center", fontsize=10)

    summary = (
        f"Water Depth: {water_depth:,.0f} ft\\n"
        f"RKB: {rkb:,.0f} ft\\n"
        f"Mudline: {mudline_md:,.0f} ft\\n"
        f"Planned TD: {td_md:,.0f} ft MD"
    )
    ax.text(right_edge-0.1, 0.015*max_depth, summary, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.4"))

    y_head = 0.06*max_depth
    ax.text((geo_left+geo_right)/2, y_head, "GEOLOGICAL TOPS", ha="center", fontweight="bold", fontsize=9)
    ax.text(well_center, y_head, "CASING / CEMENT", ha="center", fontweight="bold", fontsize=9)
    ax.text(section_x+0.7, y_head, "HOLE / MUD", ha="center", fontweight="bold", fontsize=9)
    ax.text((callout_x+right_edge)/2, y_head, "CASING DATA", ha="center", fontweight="bold", fontsize=9)

    ax.text(geo_left, mudline_md, f"Mudline  {mudline_md:,.0f} ft", fontsize=8, va="center")

    g = geocalc.dropna(subset=["MD_ft"]).sort_values("MD_ft").copy()
    for _, row in g.iterrows():
        y = float(row["MD_ft"])
        ax.plot([geo_right-0.28, geo_right], [y, y], linewidth=0.8)
        label = f'{row["Formation_Name"]}  |  MD {row["MD_ft"]:,.0f}  TVD {row["TVD_ft"]:,.0f}'
        if pd.notna(row.get("Temperature_F")):
            label += f'  |  {row["Temperature_F"]:.0f}°F'
        ax.text(geo_left, y, label, fontsize=7.4, va="center")

    c = casing.dropna(subset=["Top_MD_ft","Shoe_MD_ft"]).copy()
    sortcol = "Display_Order" if "Display_Order" in c.columns else "Shoe_MD_ft"
    c = c.sort_values(sortcol)
    widths = np.linspace(2.35, 0.9, max(len(c), 1))

    for width, (_, row) in zip(widths, c.iterrows()):
        top = float(row["Top_MD_ft"])
        shoe = float(row["Shoe_MD_ft"])
        toc = float(row["TOC_MD_ft"]) if pd.notna(row.get("TOC_MD_ft")) else top
        cement_top = max(top, toc)
        ann_width = width + 0.28
        ax.add_patch(Rectangle((-ann_width/2, cement_top), ann_width, max(shoe-cement_top, 1),
                               facecolor="0.75", alpha=0.35, edgecolor="none"))

    for width, (_, row) in zip(widths, c.iterrows()):
        top = float(row["Top_MD_ft"])
        shoe = float(row["Shoe_MD_ft"])
        od = row.get("Casing_OD_in", np.nan)
        cid = row.get("Casing_ID_in", np.nan)
        drift = row.get("Drift_in", np.nan)
        grade = row.get("Casing_Grade", "")
        hole = row.get("Hole_Size_in", np.nan)
        mwmin = row.get("Mud_Weight_Min_ppg", np.nan)
        mwmax = row.get("Mud_Weight_Max_ppg", np.nan)
        mudtype = row.get("Mud_Type", "")

        ax.add_patch(Rectangle((-width/2, top), width, max(shoe-top, 1),
                               fill=False, linewidth=1.8))
        ax.plot([-width/2, width/2], [shoe, shoe], linewidth=2.1)

        ax.plot([-width/2-0.12, -width/2], [top, top], linewidth=0.9)
        ax.text(-width/2-0.18, top, f"TOP {top:,.0f} MD", fontsize=6.8,
                ha="right", va="center")
        ax.text(width/2+0.12, shoe, f"SHOE {shoe:,.0f} MD", fontsize=6.8,
                ha="left", va="center", fontweight="bold")

        casing_txt = f'{row["String_Name"]}\\nOD {od:g}"'
        if pd.notna(cid):
            casing_txt += f' | ID {cid:g}"'
        if pd.notna(drift):
            casing_txt += f'\\nDrift {drift:g}"'
        if str(grade) and str(grade) != "nan":
            casing_txt += f' | {grade}'
        ax.text(well_center, (top+shoe)/2, casing_txt, ha="center", va="center",
                fontsize=6.8, rotation=90 if (shoe-top) > 4000 else 0)

        if pd.notna(row.get("TOC_MD_ft")):
            toc = float(row["TOC_MD_ft"])
            ax.plot([width/2, width/2+0.18], [toc, toc], linewidth=0.8)
            ax.text(width/2+0.22, toc, f"TOC {toc:,.0f}", fontsize=6.4, va="center")

        hole_txt = f'Hole {hole:g}"' if pd.notna(hole) else "Hole -"
        if pd.notna(mwmin) and pd.notna(mwmax):
            hole_txt += f'\\nMW {mwmin:.1f}-{mwmax:.1f} ppg'
        if str(mudtype) and str(mudtype) != "nan":
            hole_txt += f'\\n{mudtype}'
        ax.text(section_x, (top+shoe)/2, hole_txt, fontsize=7.0, va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7"))

    for _, row in shoes.sort_values("Shoe_MD_ft").iterrows():
        y = float(row["Shoe_MD_ft"])
        parts = [
            f'{row["String_Name"]}',
            f'Top MD {row["Casing_Top_MD_ft"]:,.0f} | Shoe MD {row["Shoe_MD_ft"]:,.0f}',
            f'Shoe TVD {row["Shoe_TVD_ft"]:,.0f}',
        ]
        if pd.notna(row.get("Casing_OD_in")):
            line = f'OD {row["Casing_OD_in"]:g}"'
            if pd.notna(row.get("Casing_ID_in")):
                line += f' | ID {row["Casing_ID_in"]:g}"'
            if pd.notna(row.get("Drift_in")):
                line += f' | Drift {row["Drift_in"]:g}"'
            parts.append(line)
        if str(row.get("Casing_Grade","")) and str(row.get("Casing_Grade","")) != "nan":
            parts.append(f'Grade {row["Casing_Grade"]}')
        parts.append(f'Inc @ shoe {row["Inc_at_Shoe_deg"]:.1f}° | Max {row["Max_Inc_to_Shoe_deg"]:.1f}°')
        if pd.notna(row.get("Temperature_F")):
            parts.append(f'Temp {row["Temperature_F"]:.0f}°F')
        if pd.notna(row.get("FIT_LOT_Value")) and str(row.get("FIT_LOT_Type","")):
            parts.append(f'{row["FIT_LOT_Type"]}: {row["FIT_LOT_Value"]:.2f} ppg EMW')
        ax.text(callout_x, y, "\\n".join(parts), fontsize=6.8, va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.55"))

    ax.text(0.65, td_md, f"TD {td_md:,.0f} ft MD", va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(geo_left-0.3, right_edge+0.2)
    ax.set_ylim(max_depth*1.055, -0.075*max_depth)
    ax.set_ylabel("Measured Depth (ft)")
    ax.set_xticks([])
    ax.grid(False)
    fig.tight_layout()
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Deepwater Wellbore Schematic Builder")
st.caption("Prototype: casing + survey + geological tops + FIT/LOT + temperature integration")

uploaded = st.file_uploader("Upload the Deepwater WBS Excel template", type=["xlsx"])

if not uploaded:
    st.info("Use the provided input template, populate your well data, then upload it here.")
    st.stop()

try:
    well, casing, survey_raw, geo, temp, fitlot = read_workbook(uploaded)
    survey = minimum_curvature_tvd(survey_raw.dropna(subset=["MD_ft","Inclination_deg","Azimuth_deg"]))
    rkb = float(well.get("RKB", 0) or 0)
    if not well.get("Mudline_TVD_RKB") or pd.isna(well.get("Mudline_TVD_RKB")):
        well["Mudline_TVD_RKB"] = float(well.get("Water_Depth",0) or 0) + rkb

    shoes = calculate_shoe_data(casing, survey, temp, well, fitlot)
    geocalc = calculate_geo_data(geo, survey, temp, rkb)

except Exception as e:
    st.error(f"Could not process workbook: {e}")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Well / Casing", "Directional Survey", "Geological Tops", "MD ↔ TVD Calculator", "Wellbore Schematic"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Well information")
        st.json({k: (None if pd.isna(v) else v) for k,v in well.items()})
    with c2:
        st.subheader("Calculated shoe / casing data")
        st.dataframe(shoes, use_container_width=True)

with tab2:
    st.subheader("Survey used by the calculation engine")
    st.dataframe(survey, use_container_width=True)
    chart_df = survey[["MD_ft","TVD_ft","Inclination_deg"]].set_index("MD_ft")
    st.line_chart(chart_df[["TVD_ft","Inclination_deg"]])

with tab3:
    st.subheader("Calculated geological tops")
    st.dataframe(geocalc, use_container_width=True)

with tab4:
    mode = st.radio("Input depth type", ["MD", "TVD"], horizontal=True)
    depth = st.number_input("Depth (ft)", min_value=0.0, value=15000.0, step=10.0)
    if mode == "MD":
        tvd = interp_by_md(survey, depth, "TVD_ft")
        inc = interp_by_md(survey, depth, "Inclination_deg")
        azi = interp_by_md(survey, depth, "Azimuth_deg")
        tempv = temp_at_tvd(temp, tvd) if pd.notna(tvd) else np.nan
        st.metric("TVD", f"{tvd:,.1f} ft" if pd.notna(tvd) else "Outside survey")
        st.write(f"Inclination: **{inc:.2f}°**" if pd.notna(inc) else "Inclination unavailable")
        st.write(f"Azimuth: **{azi:.2f}°**" if pd.notna(azi) else "Azimuth unavailable")
        st.write(f"Temperature: **{tempv:.1f} °F**" if pd.notna(tempv) else "Temperature unavailable")
    else:
        sols = md_solutions_from_tvd(survey, depth)
        if not sols:
            st.warning("No MD solution within the survey range.")
        elif len(sols) == 1:
            st.metric("MD", f"{sols[0]:,.1f} ft")
        else:
            st.warning("This TVD occurs at more than one MD. Select the applicable well section.")
            st.write([round(x,1) for x in sols])

with tab5:
    fig = draw_wbs(well, casing, shoes, geocalc, fitlot)
    st.pyplot(fig, use_container_width=False)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=220, bbox_inches="tight")
    png_buf.seek(0)

    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download WBS PNG",
            data=png_buf.getvalue(),
            file_name="wellbore_schematic.png",
            mime="image/png",
        )
    with c2:
        st.download_button(
            "Download WBS PDF",
            data=pdf_buf.getvalue(),
            file_name="wellbore_schematic.pdf",
            mime="application/pdf",
        )

out_excel = make_output_excel(well, casing, survey, geo, temp, fitlot, shoes, geocalc)
st.download_button(
    "Download calculated workbook",
    data=out_excel,
    file_name="calculated_wbs_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
