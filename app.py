
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
            "Pore_Pressure_Min_ppg": row.get("Pore_Pressure_Min_ppg", np.nan),
            "Pore_Pressure_Max_ppg": row.get("Pore_Pressure_Max_ppg", np.nan),
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

    def clean_value(value):
        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        if isinstance(value, (list, dict, tuple, set)):
            return str(value)

        if isinstance(value, np.generic):
            return value.item()

        return value

    def clean_dataframe(df):
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].map(clean_value)
        return df

    well_df = pd.DataFrame({
        "Field": [str(k) for k in well.keys()],
        "Value": [clean_value(v) for v in well.values()]
    })

    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        clean_dataframe(well_df).to_excel(
            writer, sheet_name="Well_Info", index=False
        )

        clean_dataframe(casing).to_excel(
            writer, sheet_name="Casing_Program", index=False
        )

        clean_dataframe(survey).to_excel(
            writer, sheet_name="Directional_Survey", index=False
        )

        clean_dataframe(geo).to_excel(
            writer, sheet_name="Geological_Tops", index=False
        )

        clean_dataframe(temp).to_excel(
            writer, sheet_name="Temperature_Gradient", index=False
        )

        clean_dataframe(fitlot).to_excel(
            writer, sheet_name="FIT_LOT", index=False
        )

        clean_dataframe(shoes).to_excel(
            writer, sheet_name="Calculated_Shoe_Data", index=False
        )

        clean_dataframe(geocalc).to_excel(
            writer, sheet_name="Calculated_Geo_Data", index=False
        )

    bio.seek(0)
    return bio.getvalue()


# -----------------------------
# Schematic plotting
# -----------------------------




def draw_wbs(well, casing, shoes, geocalc, fitlot=None):
    """
    Stage 5 readable WBS with:
    - fixed columns
    - larger fonts
    - logo
    - darker cement fill to TOC
    - custom casing shoe symbol
    - custom top-of-liner symbol
    - de-cluttered casing specification boxes
    """
    from matplotlib.patches import Rectangle, Polygon
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(16, 20))

    # ----------------------------
    # Helpers
    # ----------------------------
    def adjust_positions(y_values, min_sep, ymin, ymax):
        """Keep annotation centers separated so text boxes do not overlap."""
        if len(y_values) == 0:
            return []
        pairs = sorted([(float(y), i) for i, y in enumerate(y_values)], key=lambda x: x[0])
        ys = [p[0] for p in pairs]

        # forward pass
        adjusted = [ys[0]]
        for y in ys[1:]:
            adjusted.append(max(y, adjusted[-1] + min_sep))

        # if bottom overflow, shift upward
        overflow = adjusted[-1] - ymax
        if overflow > 0:
            adjusted = [y - overflow for y in adjusted]

        # backward pass to respect top bound
        if adjusted[0] < ymin:
            shift = ymin - adjusted[0]
            adjusted = [y + shift for y in adjusted]

        for i in range(len(adjusted) - 2, -1, -1):
            if adjusted[i+1] - adjusted[i] < min_sep:
                adjusted[i] = adjusted[i+1] - min_sep

        # map back
        out = [None] * len(y_values)
        for adj, (_, original_idx) in zip(adjusted, pairs):
            out[original_idx] = adj
        return out

    def draw_shoe_symbol(ax, x_left, x_right, y, h=260, color="black", wedge_color="#c8b273"):
        """
        Draw a casing shoe symbol approximating the user reference:
        black body with inward small tan wedges at the shoe.
        """
        # vertical black bodies
        bw = 0.08
        ax.add_patch(Rectangle((x_left - bw, y - h), bw, h, facecolor=color, edgecolor=color, linewidth=0))
        ax.add_patch(Rectangle((x_right, y - h), bw, h, facecolor=color, edgecolor=color, linewidth=0))

        # tan wedges pointing inward
        tw = 0.12
        ax.add_patch(Polygon([
            (x_left - tw, y - h),
            (x_left, y - h),
            (x_left, y)
        ], closed=True, facecolor=wedge_color, edgecolor=color, linewidth=0.6))
        ax.add_patch(Polygon([
            (x_right + tw, y - h),
            (x_right, y - h),
            (x_right, y)
        ], closed=True, facecolor=wedge_color, edgecolor=color, linewidth=0.6))

        # top cap at shoe depth
        ax.plot([x_left, x_right], [y, y], color=color, linewidth=1.8)

    def draw_tol_symbol(ax, x_left, x_right, y, h=280, color="black"):
        """
        Draw a top-of-liner / liner hanger style symbol approximating the user reference.
        """
        bw = 0.09
        # outer black vertical shoulders
        ax.add_patch(Rectangle((x_left - bw, y - h), bw, h, facecolor=color, edgecolor=color, linewidth=0))
        ax.add_patch(Rectangle((x_right, y - h), bw, h, facecolor=color, edgecolor=color, linewidth=0))
        # small inner hanger blocks
        inner_w = 0.06
        ax.add_patch(Rectangle((x_left + 0.02, y - h * 0.62), inner_w, h * 0.62, facecolor=color, edgecolor=color, linewidth=0))
        ax.add_patch(Rectangle((x_right - inner_w - 0.02, y - h * 0.62), inner_w, h * 0.62, facecolor=color, edgecolor=color, linewidth=0))
        # short shelf
        ax.plot([x_left - bw, x_left + 0.14], [y - h * 0.62, y - h * 0.62], color=color, linewidth=1.2)
        ax.plot([x_right - 0.14, x_right + bw], [y - h * 0.62, y - h * 0.62], color=color, linewidth=1.2)

    # ----------------------------
    # Overall scales / columns
    # ----------------------------
    depth_candidates = []
    if not casing.empty:
        depth_candidates += casing["Shoe_MD_ft"].dropna().astype(float).tolist()
    if not geocalc.empty:
        depth_candidates += geocalc["MD_ft"].dropna().astype(float).tolist()
    max_depth = max(depth_candidates) if depth_candidates else 10000.0

    rkb = float(well.get("RKB", 0) or 0)
    water_depth = float(well.get("Water_Depth", 0) or 0)
    mudline_md = float(well.get("Mudline_TVD_RKB", water_depth + rkb) or (water_depth + rkb))
    td_md = float(well.get("Planned_TD_MD", max_depth) or max_depth)
    max_depth = max(max_depth, td_md)

    x_geo = -8.5
    x_pp = -5.6
    x_well = 0.0
    x_hole = 3.1
    x_fit = 5.7
    x_casing = 7.9
    right_edge = 12.2

    y_top_margin = -0.085 * max_depth
    y_header = 0.055 * max_depth

    # ----------------------------
    # Header and logo
    # ----------------------------
    well_name = str(well.get("Well_Name", "Deepwater Well"))
    prospect = str(well.get("Field_Prospect", ""))

    ax.text(1.6, y_top_margin, "WELLBORE DIAGRAM", ha="center",
            fontsize=18, fontweight="bold")
    ax.text(1.6, y_top_margin + 0.028 * max_depth, well_name, ha="center",
            fontsize=13, fontweight="bold")
    if prospect and prospect != "nan":
        ax.text(1.6, y_top_margin + 0.050 * max_depth, prospect, ha="center",
                fontsize=10)

    # Logo in top-left if present
    logo_path = Path(__file__).with_name("eni_logo.png")
    if logo_path.exists():
        try:
            logo = plt.imread(str(logo_path))
            # x0, x1, y0, y1 in data coordinates
            ax.imshow(
                logo,
                extent=[-10.0, -7.7, y_top_margin + 0.070 * max_depth, y_top_margin - 0.010 * max_depth],
                aspect="auto",
                zorder=0,
            )
        except Exception:
            pass

    headers = [
        (x_geo, "GEOLOGICAL TOPS"),
        (x_pp, "PORE PRESSURE"),
        (x_well, "WELLBORE"),
        (x_hole, "HOLE / MUD"),
        (x_fit, "FIT / LOT"),
        (x_casing, "CASING SPECS"),
    ]
    for x, txt in headers:
        ax.text(x, y_header, txt, ha="center", va="center", fontsize=9.5, fontweight="bold")

    separators = [-6.8, -4.1, 1.9, 4.5, 6.7]
    for x in separators:
        ax.vlines(x, 0.072 * max_depth, max_depth * 1.012, linewidth=0.65, alpha=0.45)

    summary = (
        f"Water Depth: {water_depth:,.0f} ft\n"
        f"RKB: {rkb:,.0f} ft\n"
        f"Mudline: {mudline_md:,.0f} ft\n"
        f"TD: {td_md:,.0f} ft MD"
    )
    ax.text(right_edge, 0.005 * max_depth, summary, ha="right", va="top", fontsize=8.5)

    # mudline local mark
    ax.text(x_geo - 0.4, mudline_md, f"Mudline\n{mudline_md:,.0f} MD", va="center", ha="center", fontsize=8)
    ax.plot([-0.9, 0.9], [mudline_md, mudline_md], color="black", linewidth=1.2)

    # ----------------------------
    # Geological tops
    # ----------------------------
    g = geocalc.dropna(subset=["MD_ft"]).sort_values("MD_ft")
    geo_positions = [float(y) for y in g["MD_ft"].tolist()]
    geo_adj = adjust_positions(geo_positions, min_sep=700, ymin=0.09 * max_depth, ymax=max_depth * 0.98)

    for (_, row), y_text in zip(g.iterrows(), geo_adj):
        y_actual = float(row["MD_ft"])
        # local depth tick
        ax.plot([-6.95, -6.8], [y_actual, y_actual], color="black", linewidth=0.8)
        txt = f'{row["Formation_Name"]}\nMD {row["MD_ft"]:,.0f} | TVD {row["TVD_ft"]:,.0f}'
        ax.text(x_geo, y_text, txt, ha="center", va="center", fontsize=7.6)

    # ----------------------------
    # Wellbore / casing
    # ----------------------------
    c = casing.dropna(subset=["Top_MD_ft", "Shoe_MD_ft"]).copy()
    sortcol = "Display_Order" if "Display_Order" in c.columns else "Shoe_MD_ft"
    c = c.sort_values(sortcol)
    widths = np.linspace(2.25, 0.78, max(len(c), 1))

    for width, (_, row) in zip(widths, c.iterrows()):
        top = float(row["Top_MD_ft"])
        shoe = float(row["Shoe_MD_ft"])
        toc = float(row["TOC_MD_ft"]) if pd.notna(row.get("TOC_MD_ft")) else top
        ann_width = width + 0.24

        # dark gray cement annulus from TOC to shoe
        cement_top = max(top, toc)
        ax.add_patch(Rectangle(
            (-ann_width / 2, cement_top),
            ann_width,
            max(shoe - cement_top, 1),
            facecolor="dimgray",
            alpha=0.55,
            edgecolor="none"
        ))

        # casing outline
        ax.add_patch(Rectangle(
            (-width / 2, top),
            width,
            max(shoe - top, 1),
            fill=False,
            linewidth=1.8
        ))

        # casing OD label only
        od = row.get("Casing_OD_in", np.nan)
        label = f'{od:g}"' if pd.notna(od) else str(row["String_Name"])
        ax.text(0, (top + shoe) / 2, label, ha="center", va="center", fontsize=7.0)

        x_left = -width / 2
        x_right = width / 2

        # liner top symbol if applicable
        is_liner = "liner" in str(row.get("String_Name", "")).lower() or "liner" in str(row.get("Casing_Top_Type", "")).lower()
        if is_liner:
            draw_tol_symbol(ax, x_left, x_right, top, h=300)
            ax.text(x_left - 0.35, top, f"TOL {top:,.0f}", ha="right", va="center", fontsize=7.4, fontweight="bold")
        else:
            # standard top depth text, small and outside casing
            ax.text(x_left - 0.28, top, f"Top {top:,.0f}", ha="right", va="center", fontsize=7.1)

        # TOC short marker
        if pd.notna(row.get("TOC_MD_ft")):
            toc_md = float(row["TOC_MD_ft"])
            ax.plot([x_right, x_right + 0.18], [toc_md, toc_md], color="black", linewidth=0.8)
            ax.text(x_right + 0.24, toc_md, f"TOC {toc_md:,.0f}", va="center", fontsize=7.0)

        # custom shoe symbol
        draw_shoe_symbol(ax, x_left, x_right, shoe, h=260)
        ax.text(x_right + 0.30, shoe, f"Shoe {shoe:,.0f} MD", ha="left", va="center",
                fontsize=7.4, fontweight="bold")

    # ----------------------------
    # Column entries per section
    # ----------------------------
    shoes_sorted = shoes.sort_values("Shoe_MD_ft").reset_index(drop=True).copy()
    shoe_actual = shoes_sorted["Shoe_MD_ft"].astype(float).tolist()
    right_adj = adjust_positions(shoe_actual, min_sep=1800, ymin=0.11 * max_depth, ymax=max_depth * 0.96)
    fit_adj = adjust_positions(shoe_actual, min_sep=1200, ymin=0.11 * max_depth, ymax=max_depth * 0.96)
    hole_mid = [(float(t) + float(s)) / 2 for t, s in zip(shoes_sorted["Casing_Top_MD_ft"], shoes_sorted["Shoe_MD_ft"])]

    for idx, row in shoes_sorted.iterrows():
        top = float(row["Casing_Top_MD_ft"])
        shoe = float(row["Shoe_MD_ft"])
        mid = hole_mid[idx]

        # Pore pressure column, centered per interval
        ppmin = row.get("Pore_Pressure_Min_ppg", np.nan)
        ppmax = row.get("Pore_Pressure_Max_ppg", np.nan)
        if pd.notna(ppmin) and pd.notna(ppmax):
            pp_text = f"{ppmin:.1f}-{ppmax:.1f}\nppg"
        elif pd.notna(ppmax):
            pp_text = f"{ppmax:.1f}\nppg"
        else:
            pp_text = "-"
        ax.text(x_pp, mid, pp_text, ha="center", va="center", fontsize=7.8)

        # Hole / Mud column, centered per interval
        hole = row.get("Hole_Size_in", np.nan)
        mwmin = row.get("Mud_Weight_Min_ppg", np.nan)
        mwmax = row.get("Mud_Weight_Max_ppg", np.nan)
        mudtype = str(row.get("Mud_Type", ""))
        hole_lines = []
        if pd.notna(hole):
            hole_lines.append(f'Hole {hole:g}"')
        if pd.notna(mwmin) and pd.notna(mwmax):
            hole_lines.append(f"MW {mwmin:.1f}-{mwmax:.1f} ppg")
        if mudtype and mudtype != "nan":
            hole_lines.append(mudtype)
        ax.text(x_hole, mid, "\n".join(hole_lines), ha="center", va="center", fontsize=7.8)

        # FIT / LOT column, compact box near shoe but de-cluttered
        fit_type = str(row.get("FIT_LOT_Type", ""))
        fit_val = row.get("FIT_LOT_Value", np.nan)
        fit_text = "-"
        if fit_type and fit_type != "nan" and pd.notna(fit_val):
            fit_text = f"{fit_type}\n{fit_val:.2f} ppg"
        ax.text(x_fit, fit_adj[idx], fit_text, ha="center", va="center", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.55"))

        # Casing specs box, larger font and fewer cluttering overlaps
        od = row.get("Casing_OD_in", np.nan)
        cid = row.get("Casing_ID_in", np.nan)
        drift = row.get("Drift_in", np.nan)
        grade = str(row.get("Casing_Grade", ""))
        temp = row.get("Temperature_F", np.nan)
        inc = row.get("Inc_at_Shoe_deg", np.nan)
        max_inc = row.get("Max_Inc_to_Shoe_deg", np.nan)
        shoe_tvd = row.get("Shoe_TVD_ft", np.nan)

        spec_lines = [str(row["String_Name"])]
        size_bits = []
        if pd.notna(od):
            size_bits.append(f'OD {od:g}"')
        if pd.notna(cid):
            size_bits.append(f'ID {cid:g}"')
        if size_bits:
            spec_lines.append(" | ".join(size_bits))
        if pd.notna(drift):
            spec_lines.append(f'Drift {drift:g}"')
        if grade and grade != "nan":
            spec_lines.append(f"Grade {grade}")
        spec_lines.append(f"Top {top:,.0f} MD")
        if pd.notna(shoe_tvd):
            spec_lines.append(f"Shoe {shoe:,.0f} MD / {shoe_tvd:,.0f} TVD")
        else:
            spec_lines.append(f"Shoe {shoe:,.0f} MD")
        if pd.notna(inc) and pd.notna(max_inc):
            spec_lines.append(f"Inc {inc:.1f}° | Max {max_inc:.1f}°")
        if pd.notna(temp):
            spec_lines.append(f"Temp {temp:.0f}°F")

        ax.text(
            x_casing, right_adj[idx], "\n".join(spec_lines),
            ha="left", va="center", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.55")
        )

    # TD marker
    ax.text(0.9, td_md, f"TD {td_md:,.0f} MD", va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlim(-10.6, right_edge + 0.4)
    ax.set_ylim(max_depth * 1.04, y_top_margin - 0.02 * max_depth)
    ax.set_ylabel("Measured Depth (ft)", fontsize=11)
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
        st.subheader("Calculated hole-section / casing data")
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


try:
    out_excel = make_output_excel(
        well, casing, survey, geo, temp, fitlot, shoes, geocalc
    )

    st.download_button(
        "Download calculated workbook",
        data=out_excel,
        file_name="calculated_wbs_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

except Exception as e:
    st.warning(
        "The WBS was generated successfully, but the calculated Excel "
        "workbook could not be created."
    )
    st.error(str(e))

