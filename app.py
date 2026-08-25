
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
        toc_md = float(row["TOC_MD_ft"]) if pd.notna(row.get("TOC_MD_ft")) else np.nan
        toc_tvd = interp_by_md(survey, toc_md, "TVD_ft") if pd.notna(toc_md) else np.nan

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
            "TOC_MD_ft": toc_md,
            "TOC_TVD_ft": toc_tvd,
            "Shoe_MD_ft": md,
            "Shoe_TVD_ft": tvd,
            "Shoe_TVDSS_ft": tvd - rkb if pd.notna(tvd) else np.nan,
            "Inc_at_Shoe_deg": inc,
            "Max_Inc_to_Shoe_deg": max_inc,
            "Azimuth_at_Shoe_deg": azi,
            "Temperature_F": temperature,
            "FIT_LOT_Type": test_type,
            "FIT_LOT_Value": test_val,
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
    Stage 10:
    - start illustration 1000 ft above mudline
    - display Surface and Bottom Hole XY coordinates in the header
    - keep combined Hole/Mud and Casing/FIT-LOT layout
    """
    from matplotlib.patches import Rectangle, Polygon
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(16, 20))

    def adjust_positions(y_values, min_sep, ymin, ymax):
        if len(y_values) == 0:
            return []
        pairs = sorted([(float(y), i) for i, y in enumerate(y_values)], key=lambda x: x[0])
        ys = [p[0] for p in pairs]
        adjusted = [max(ys[0], ymin)]
        for y in ys[1:]:
            adjusted.append(max(y, adjusted[-1] + min_sep))
        overflow = adjusted[-1] - ymax
        if overflow > 0:
            adjusted = [y - overflow for y in adjusted]
        if adjusted[0] < ymin:
            shift = ymin - adjusted[0]
            adjusted = [y + shift for y in adjusted]
        for i in range(len(adjusted) - 2, -1, -1):
            if adjusted[i+1] - adjusted[i] < min_sep:
                adjusted[i] = adjusted[i+1] - min_sep
        out = [None] * len(y_values)
        for adj, (_, original_idx) in zip(adjusted, pairs):
            out[original_idx] = adj
        return out

    def draw_shoe_symbol(ax, x_left, x_right, y, h=250, color='black', wedge_color='#c8b273'):
        bw = 0.08
        ax.add_patch(Rectangle((x_left - bw, y - h), bw, h, facecolor=color, edgecolor=color, linewidth=0))
        ax.add_patch(Rectangle((x_right, y - h), bw, h, facecolor=color, edgecolor=color, linewidth=0))
        tw = 0.12
        ax.add_patch(Polygon([(x_left, y - h), (x_left - tw, y), (x_left, y)], closed=True,
                             facecolor=wedge_color, edgecolor=color, linewidth=0.6))
        ax.add_patch(Polygon([(x_right, y - h), (x_right + tw, y), (x_right, y)], closed=True,
                             facecolor=wedge_color, edgecolor=color, linewidth=0.6))
        # No horizontal line across the casing at the shoe.

    def draw_tol_symbol(ax, liner_left, liner_right, outer_left, outer_right, y, h=300, color='black'):
        """
        Liner-hanger / TOL symbol matching the reference:
        compact stepped black blocks bridge the annular gap from the parent casing
        directly to the liner wall on both sides. The liner continues below the hanger.
        """
        hanger_h = h * 0.30
        hanger_top = y - hanger_h * 0.72
        hanger_bottom = hanger_top + hanger_h

        # Small overlap removes any white hairline between black fill and casing/liner walls.
        overlap = 0.025

        # Left bridge: parent casing -> liner.
        left_x0 = outer_left - overlap
        left_x1 = liner_left + overlap
        ax.add_patch(Rectangle(
            (left_x0, hanger_top),
            max(left_x1 - left_x0, 0.03),
            hanger_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=7
        ))

        # Right bridge: liner -> parent casing.
        right_x0 = liner_right - overlap
        right_x1 = outer_right + overlap
        ax.add_patch(Rectangle(
            (right_x0, hanger_top),
            max(right_x1 - right_x0, 0.03),
            hanger_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=7
        ))

        # Stepped upper shoulders near the liner, like the reference snapshot.
        step_h = hanger_h * 0.30
        step_w = min(max((liner_left - outer_left) * 0.34, 0.06), 0.16)
        ax.add_patch(Rectangle(
            (liner_left - step_w, hanger_top - step_h * 0.18),
            step_w + overlap,
            step_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=8
        ))
        ax.add_patch(Rectangle(
            (liner_right - overlap, hanger_top - step_h * 0.18),
            step_w + overlap,
            step_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=8
        ))

        # Short black liner continuation below the hanger so the transition is visually solid.
        liner_bar_w = 0.055
        lower_h = h * 0.34
        ax.add_patch(Rectangle(
            (liner_left - liner_bar_w / 2, hanger_bottom - overlap),
            liner_bar_w, lower_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=8
        ))
        ax.add_patch(Rectangle(
            (liner_right - liner_bar_w / 2, hanger_bottom - overlap),
            liner_bar_w, lower_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=8
        ))

    depth_candidates = []
    if not casing.empty:
        depth_candidates += casing['Shoe_MD_ft'].dropna().astype(float).tolist()
        depth_candidates += casing['Top_MD_ft'].dropna().astype(float).tolist()
    if not geocalc.empty:
        depth_candidates += geocalc['MD_ft'].dropna().astype(float).tolist()

    max_depth = max(depth_candidates) if depth_candidates else 10000.0
    rkb = float(well.get('RKB', 0) or 0)
    water_depth = float(well.get('Water_Depth', 0) or 0)
    mudline_md = float(well.get('Mudline_TVD_RKB', water_depth + rkb) or (water_depth + rkb))
    td_md = float(well.get('Planned_TD_MD', max_depth) or max_depth)
    max_depth = max(max_depth, td_md)

    # Begin the visible schematic 1500 ft above mudline
    visible_top = max(0.0, mudline_md - 1500.0)

    # Header positions placed above the visible top
    # Keep the well name comfortably inside the frame and move column headers a little higher.
    title_y = visible_top - 620
    subtitle_y = visible_top - 410
    prospect_y = visible_top - 250
    coord_y = visible_top - 80
    # Put column headers a little above the mudline
    header_y = mudline_md - 500

    x_geo = -9.0
    x_pp = -6.2
    x_hole = -3.0
    x_well = 0.4
    x_info = 6.0
    left_edge = -11.3
    right_edge = 11.7

    well_name = str(well.get('Well_Name', 'Deepwater Well'))
    prospect = str(well.get('Field_Prospect', ''))
    surf_x = well.get('Surface_X', '')
    surf_y = well.get('Surface_Y', '')
    bh_x = well.get('BottomHole_X', '')
    bh_y = well.get('BottomHole_Y', '')

    ax.text(0, title_y, well_name, ha='center', fontsize=14, fontweight='bold')
    if prospect and prospect != 'nan':
        ax.text(0, subtitle_y, prospect, ha='center', fontsize=10)

    coord_lines = []
    if surf_x != '' or surf_y != '':
        coord_lines.append(f'Surface Location: X={surf_x}, Y={surf_y}')
    if bh_x != '' or bh_y != '':
        coord_lines.append(f'Bottom Hole Location: X={bh_x}, Y={bh_y}')
    if coord_lines:
        # Place the coordinates box to the right of the Eni logo
        coord_box_x = left_edge + 4.6
        ax.text(coord_box_x, coord_y, '\n'.join(coord_lines), ha='left', va='top', fontsize=8.8,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='0.55'))

    logo_path = Path(__file__).with_name('eni_logo.png')
    if logo_path.exists():
        try:
            logo = plt.imread(str(logo_path))
            ax.imshow(logo, extent=[left_edge + 0.5, left_edge + 2.8, visible_top - 40, visible_top - 760],
                      aspect='auto', zorder=0)
        except Exception:
            pass

    headers = [
        (x_geo, 'GEOLOGICAL TOPS'),
        (x_pp, 'PORE PRESSURE'),
        (x_hole, 'MUD'),
        (x_well, 'WELLBORE'),
        (x_info, 'HOLE / CASING / FIT-LOT'),
    ]
    for x, txt in headers:
        ax.text(x, header_y, txt, ha='center', va='center', fontsize=9.5, fontweight='bold')

    separators = [-7.6, -4.8, -1.4, 2.2]
    for x in separators:
        ax.vlines(x, header_y + 90, max_depth * 1.012, linewidth=0.65, alpha=0.45)

    summary = (
        f'Water Depth: {water_depth:,.0f} ft\n'
        f'RKB: {rkb:,.0f} ft\n'
        f'Mudline: {mudline_md:,.0f} ft\n'
        f'TD: {td_md:,.0f} ft MD'
    )
    ax.text(right_edge, visible_top - 760, summary, ha='right', va='top', fontsize=8.5)

    ax.text(x_geo - 0.1, mudline_md, f'Mudline\n{mudline_md:,.0f} MD', va='center', ha='center', fontsize=8)
    ax.plot([x_well - 1.05, x_well + 1.05], [mudline_md, mudline_md], color='black', linewidth=1.2)

    g = geocalc.dropna(subset=['MD_ft']).sort_values('MD_ft')
    geo_positions = [float(y) for y in g['MD_ft'].tolist()]
    geo_adj = adjust_positions(geo_positions, min_sep=700, ymin=mudline_md + 250, ymax=max_depth * 0.98)
    for (_, row), y_text in zip(g.iterrows(), geo_adj):
        y_actual = float(row['MD_ft'])
        ax.plot([-7.75, -7.6], [y_actual, y_actual], color='black', linewidth=0.8)
        txt = f"{row['Formation_Name']}\nMD {row['MD_ft']:,.0f} | TVD {row['TVD_ft']:,.0f}"
        ax.text(x_geo, y_text, txt, ha='center', va='center', fontsize=7.6)

    c = casing.dropna(subset=['Top_MD_ft', 'Shoe_MD_ft']).copy()
    sortcol = 'Display_Order' if 'Display_Order' in c.columns else 'Shoe_MD_ft'
    c = c.sort_values(sortcol)
    widths = np.linspace(2.8, 1.0, max(len(c), 1))

    casing_rows = list(c.iterrows())
    for idx_casing, (width, (_, row)) in enumerate(zip(widths, casing_rows)):
        top = float(row['Top_MD_ft'])
        shoe = float(row['Shoe_MD_ft'])
        toc_md = float(row['TOC_MD_ft']) if pd.notna(row.get('TOC_MD_ft')) else np.nan

        cement_top = max(top, toc_md) if pd.notna(toc_md) else top
        annulus_outer = width + 0.28
        ann_gap = (annulus_outer - width) / 2.0
        # Slight overlap so the gray fill visually touches the casing walls cleanly.
        overlap = 0.02
        if shoe > cement_top:
            ax.add_patch(Rectangle((x_well - annulus_outer / 2, cement_top), ann_gap + overlap, shoe - cement_top,
                                   facecolor='dimgray', alpha=0.70, edgecolor='none'))
            ax.add_patch(Rectangle((x_well + width / 2 - overlap, cement_top), ann_gap + overlap, shoe - cement_top,
                                   facecolor='dimgray', alpha=0.70, edgecolor='none'))

        # Draw only the vertical casing sides so no horizontal line appears at the shoe depth.
        x_left = x_well - width / 2
        x_right = x_well + width / 2
        ax.plot([x_left, x_left], [top, shoe], color='black', linewidth=1.9)
        ax.plot([x_right, x_right], [top, shoe], color='black', linewidth=1.9)
        is_liner = 'liner' in str(row.get('String_Name', '')).lower() or 'liner' in str(row.get('Casing_Top_Type', '')).lower()
        if is_liner:
            # Use the previous wider casing as the parent casing for the liner hanger.
            if idx_casing > 0:
                parent_width = float(widths[idx_casing - 1])
            else:
                parent_width = float(width) + 0.55
            outer_left = x_well - parent_width / 2
            outer_right = x_well + parent_width / 2
            draw_tol_symbol(ax, x_left, x_right, outer_left, outer_right, top, h=300)
        if pd.notna(toc_md):
            ax.plot([x_right, x_right + 0.15], [toc_md, toc_md], color='black', linewidth=0.8)
        draw_shoe_symbol(ax, x_left, x_right, shoe, h=250)

    shoes_sorted = shoes.sort_values('Shoe_MD_ft').reset_index(drop=True).copy()
    shoe_actual = shoes_sorted['Shoe_MD_ft'].astype(float).tolist()
    info_adj = adjust_positions(shoe_actual, min_sep=2100, ymin=mudline_md + 500, ymax=max_depth * 0.96)
    # Place each hole-section PP and Mud annotation just below the previous shoe.
    # For the first hole section, use the mudline as the starting reference.
    section_label_y = []
    previous_shoe = mudline_md
    for idx, row in shoes_sorted.iterrows():
        current_shoe = float(row['Shoe_MD_ft'])
        section_length = max(current_shoe - previous_shoe, 1.0)
        offset = min(max(section_length * 0.12, 220.0), 650.0)
        section_label_y.append(previous_shoe + offset)
        previous_shoe = current_shoe

    for idx, row in shoes_sorted.iterrows():
        top = float(row['Casing_Top_MD_ft'])
        shoe = float(row['Shoe_MD_ft'])
        section_y = section_label_y[idx]
        toc_md = row.get('TOC_MD_ft', np.nan)
        toc_tvd = row.get('TOC_TVD_ft', np.nan)
        shoe_tvd = row.get('Shoe_TVD_ft', np.nan)
        top_tvd = row.get('Casing_Top_TVD_ft', np.nan)

        ppmin = row.get('Pore_Pressure_Min_ppg', np.nan)
        ppmax = row.get('Pore_Pressure_Max_ppg', np.nan)
        if pd.notna(ppmin) and pd.notna(ppmax):
            pp_text = f'{ppmin:.1f}-{ppmax:.1f}\nppg'
        elif pd.notna(ppmax):
            pp_text = f'{ppmax:.1f}\nppg'
        else:
            pp_text = '-'
        ax.text(x_pp, section_y, pp_text, ha='center', va='top', fontsize=7.8)

        hole = row.get('Hole_Size_in', np.nan)
        mwmin = row.get('Mud_Weight_Min_ppg', np.nan)
        mwmax = row.get('Mud_Weight_Max_ppg', np.nan)
        mudtype = str(row.get('Mud_Type', ''))
        mud_lines = []
        if pd.notna(mwmin) and pd.notna(mwmax):
            mud_lines.append(f'MW {mwmin:.1f}-{mwmax:.1f} ppg')
        if mudtype and mudtype != 'nan':
            mud_lines.append(mudtype)
        ax.text(x_hole, section_y, '\n'.join(mud_lines), ha='center', va='top', fontsize=7.8)

        od = row.get('Casing_OD_in', np.nan)
        cid = row.get('Casing_ID_in', np.nan)
        drift = row.get('Drift_in', np.nan)
        grade = str(row.get('Casing_Grade', ''))
        temp = row.get('Temperature_F', np.nan)
        inc = row.get('Inc_at_Shoe_deg', np.nan)
        max_inc = row.get('Max_Inc_to_Shoe_deg', np.nan)
        fit_type = str(row.get('FIT_LOT_Type', ''))
        fit_val = row.get('FIT_LOT_Value', np.nan)

        spec_lines = []
        if pd.notna(hole):
            spec_lines.append(f'Hole {hole:g}"')
        spec_lines.append(str(row['String_Name']))
        size_bits = []
        if pd.notna(od):
            size_bits.append(f'OD {od:g}"')
        if pd.notna(cid):
            size_bits.append(f'ID {cid:g}"')
        if size_bits:
            spec_lines.append(' | '.join(size_bits))
        if pd.notna(drift):
            spec_lines.append(f'Drift {drift:g}"')
        if grade and grade != 'nan':
            spec_lines.append(f'Grade {grade}')
        if pd.notna(shoe_tvd):
            spec_lines.append(f'Shoe {shoe:,.0f} MD / {shoe_tvd:,.0f} TVD')
        else:
            spec_lines.append(f'Shoe {shoe:,.0f} MD')
        if pd.notna(top_tvd):
            prefix = 'TOL' if 'liner' in str(row['String_Name']).lower() else 'Top'
            spec_lines.append(f'{prefix} {top:,.0f} MD / {top_tvd:,.0f} TVD')
        else:
            spec_lines.append(f'Top {top:,.0f} MD')
        if pd.notna(toc_md) and pd.notna(toc_tvd):
            spec_lines.append(f'TOC {toc_md:,.0f} MD / {toc_tvd:,.0f} TVD')
        elif pd.notna(toc_md):
            spec_lines.append(f'TOC {toc_md:,.0f} MD')
        if fit_type and fit_type != 'nan' and pd.notna(fit_val):
            spec_lines.append(f'{fit_type} {fit_val:.2f} ppg EMW')
        if pd.notna(inc) and pd.notna(max_inc):
            spec_lines.append(f'Inc {inc:.1f}° | Max {max_inc:.1f}°')
        if pd.notna(temp):
            spec_lines.append(f'Temp {temp:.0f}°F')

        ax.text(x_info, info_adj[idx], '\n'.join(spec_lines), ha='left', va='center', fontsize=7.7,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='0.55'))

    ax.text(x_well + 1.0, td_md, f'TD {td_md:,.0f} MD', va='center', fontsize=8.5, fontweight='bold')

    ax.set_xlim(left_edge, right_edge)
    ax.set_ylim(max_depth * 1.04, visible_top - 900)
    ax.set_ylabel('Measured Depth (ft)', fontsize=11)
    ax.set_xticks([])
    ax.grid(False)
    fig.tight_layout()
    return fig


def draw_directional_wbs(well, survey, shoes, geocalc):
    """
    Directional WBS side-view based on the survey trajectory.
    X-axis = horizontal displacement from surface.
    Y-axis = TVD.
    """
    fig, ax = plt.subplots(figsize=(14, 18))

    s = survey.dropna(subset=['MD_ft', 'TVD_ft']).sort_values('MD_ft').reset_index(drop=True).copy()
    if s.empty:
        ax.text(0.5, 0.5, 'No valid directional survey data.', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    # Horizontal displacement for a clean side-view trajectory.
    north = s['Northing_calc_ft'].to_numpy(float) if 'Northing_calc_ft' in s.columns else np.zeros(len(s))
    east = s['Easting_calc_ft'].to_numpy(float) if 'Easting_calc_ft' in s.columns else np.zeros(len(s))
    s['HD_ft'] = np.sqrt(north**2 + east**2)

    md_arr = s['MD_ft'].to_numpy(float)
    tvd_arr = s['TVD_ft'].to_numpy(float)
    hd_arr = s['HD_ft'].to_numpy(float)

    def interp_on_survey(md, col):
        if pd.isna(md):
            return np.nan
        if md < md_arr.min() or md > md_arr.max():
            return np.nan
        if col == 'HD_ft':
            return float(np.interp(md, md_arr, hd_arr))
        if col == 'TVD_ft':
            return float(np.interp(md, md_arr, tvd_arr))
        if col in s.columns:
            return float(np.interp(md, md_arr, s[col].to_numpy(float)))
        return np.nan

    def segment_between(md1, md2):
        if pd.isna(md1) or pd.isna(md2):
            return np.array([]), np.array([])
        lo, hi = sorted([float(md1), float(md2)])
        md_points = [lo]
        internal = s[(s['MD_ft'] > lo) & (s['MD_ft'] < hi)]['MD_ft'].tolist()
        md_points.extend(internal)
        md_points.append(hi)
        md_points = np.array(sorted(set(float(x) for x in md_points)))
        x = np.interp(md_points, md_arr, hd_arr)
        y = np.interp(md_points, md_arr, tvd_arr)
        return x, y

    # Header
    well_name = str(well.get('Well_Name', 'Directional WBS'))
    prospect = str(well.get('Field_Prospect', ''))
    surf_x = well.get('Surface_X', '')
    surf_y = well.get('Surface_Y', '')
    bh_x = well.get('BottomHole_X', '')
    bh_y = well.get('BottomHole_Y', '')

    title = well_name if well_name and well_name != 'nan' else 'Directional WBS'
    ax.set_title(f'{title} - Directional WBS', fontsize=16, fontweight='bold', pad=18)
    if prospect and prospect != 'nan':
        ax.text(0.5, 1.005, str(prospect), transform=ax.transAxes, ha='center', va='bottom', fontsize=10)

    header_lines = []
    if surf_x != '' or surf_y != '':
        header_lines.append(f'Surface: X={surf_x}, Y={surf_y}')
    if bh_x != '' or bh_y != '':
        header_lines.append(f'Bottom Hole: X={bh_x}, Y={bh_y}')
    if header_lines:
        ax.text(0.98, 1.02, '\n'.join(header_lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='0.55'))

    # Base trajectory
    ax.plot(hd_arr, tvd_arr, color='0.80', linewidth=2.2, zorder=1)

    # Plot casing strings as thicker trajectory segments.
    csort = shoes.sort_values('Shoe_MD_ft').reset_index(drop=True).copy()
    if not csort.empty:
        line_widths = np.linspace(9.5, 3.5, len(csort))
        colors = ['black', '0.15', '0.25', '0.35', '0.45', '0.55']
        label_offsets = np.linspace(0, 180, len(csort))
        for i, (_, row) in enumerate(csort.iterrows()):
            top_md = row.get('Casing_Top_MD_ft', np.nan)
            shoe_md = row.get('Shoe_MD_ft', np.nan)
            x_seg, y_seg = segment_between(top_md, shoe_md)
            if len(x_seg) >= 2:
                ax.plot(x_seg, y_seg, color=colors[min(i, len(colors)-1)], linewidth=float(line_widths[i]),
                        solid_capstyle='round', zorder=2)
                # overlay a lighter inner line to make the string visible as a track
                inner_lw = max(float(line_widths[i]) - 4.0, 1.0)
                ax.plot(x_seg, y_seg, color='white', linewidth=inner_lw, alpha=0.55,
                        solid_capstyle='round', zorder=3)

            shoe_x = interp_on_survey(shoe_md, 'HD_ft')
            shoe_y = interp_on_survey(shoe_md, 'TVD_ft')
            top_tvd = row.get('Casing_Top_TVD_ft', np.nan)
            shoe_tvd = row.get('Shoe_TVD_ft', np.nan)
            if pd.notna(shoe_x) and pd.notna(shoe_y):
                ax.scatter([shoe_x], [shoe_y], marker='v', s=90, color='#c8b273', edgecolor='black', zorder=5)
                info = [str(row.get('String_Name', 'String'))]
                if pd.notna(shoe_md) and pd.notna(shoe_tvd):
                    info.append(f'Shoe {shoe_md:,.0f} MD / {shoe_tvd:,.0f} TVD')
                elif pd.notna(shoe_md):
                    info.append(f'Shoe {shoe_md:,.0f} MD')
                prefix = 'TOL' if 'liner' in str(row.get('String_Name','')).lower() else 'Top'
                if pd.notna(top_md) and pd.notna(top_tvd):
                    info.append(f'{prefix} {top_md:,.0f} MD / {top_tvd:,.0f} TVD')
                inc = row.get('Inc_at_Shoe_deg', np.nan)
                if pd.notna(inc):
                    info.append(f'Inc {inc:.1f}°')
                dx = 60 if shoe_x <= np.nanmax(hd_arr) * 0.65 else -220
                ha = 'left' if dx > 0 else 'right'
                dy = float(label_offsets[i])
                ax.annotate('\n'.join(info), xy=(shoe_x, shoe_y), xytext=(shoe_x + dx, shoe_y + dy),
                            textcoords='data', ha=ha, va='center', fontsize=7.6,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.6'),
                            arrowprops=dict(arrowstyle='-', color='0.5', linewidth=0.8), zorder=6)

    # Geological tops along trajectory.
    if geocalc is not None and not geocalc.empty:
        g = geocalc.dropna(subset=['MD_ft', 'TVD_ft']).sort_values('MD_ft').copy()
        for _, row in g.iterrows():
            md = float(row['MD_ft'])
            gx = interp_on_survey(md, 'HD_ft')
            gy = float(row['TVD_ft'])
            if pd.notna(gx) and pd.notna(gy):
                ax.plot([gx - 28, gx + 28], [gy, gy], color='tab:red', linewidth=1.0, zorder=4)
                ax.text(gx + 35, gy, f"{row['Formation_Name']}\nMD {md:,.0f}", fontsize=7.0,
                        va='center', ha='left', color='black', zorder=4)

    # Surface / TD markers
    ax.scatter([hd_arr[0]], [tvd_arr[0]], color='tab:blue', s=40, zorder=6)
    ax.text(hd_arr[0], tvd_arr[0], ' Surface', fontsize=8, va='bottom', ha='left')
    ax.scatter([hd_arr[-1]], [tvd_arr[-1]], color='tab:green', s=40, zorder=6)
    ax.text(hd_arr[-1], tvd_arr[-1], ' TD', fontsize=8, va='bottom', ha='left')

    max_inc = float(s['Inclination_deg'].max()) if 'Inclination_deg' in s.columns and not s.empty else np.nan
    final_hd = float(hd_arr[-1]) if len(hd_arr) else np.nan
    summary = []
    if pd.notna(max_inc):
        summary.append(f'Max inclination: {max_inc:.1f}°')
    if pd.notna(final_hd):
        summary.append(f'Final horizontal displacement: {final_hd:,.0f} ft')
    if summary:
        ax.text(0.02, 0.02, '\n'.join(summary), transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='0.55'))

    xmin, xmax = float(np.nanmin(hd_arr)), float(np.nanmax(hd_arr))
    xspan = max(xmax - xmin, 1.0)
    ax.set_xlim(xmin - 0.12 * xspan - 50, xmax + 0.25 * xspan + 150)
    ax.set_ylim(float(np.nanmax(tvd_arr)) * 1.03, max(0.0, float(np.nanmin(tvd_arr)) - 250))
    ax.set_xlabel('Horizontal displacement (ft)')
    ax.set_ylabel('True Vertical Depth (ft)')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig



def draw_3d_directional_wbs(
    well,
    survey,
    shoes,
    geocalc,
    show_trajectory=True,
    show_casing=True,
    show_shoes=True,
    show_shoe_labels=True,
    show_geology=True,
    show_surface_td=True,
    show_coordinate_box=True,
    view_elev=24,
    view_azim=-58,
):
    """3D directional WBS using Easting, Northing and TVD with layer toggles."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 16))
    ax = fig.add_subplot(111, projection='3d')

    s = survey.dropna(subset=['MD_ft', 'TVD_ft']).sort_values('MD_ft').reset_index(drop=True).copy()
    if s.empty:
        ax.text2D(0.5, 0.5, 'No valid directional survey data.', transform=ax.transAxes,
                  ha='center', va='center')
        return fig

    md_arr = s['MD_ft'].to_numpy(float)
    tvd_arr = s['TVD_ft'].to_numpy(float)
    north_arr = s['Northing_calc_ft'].to_numpy(float) if 'Northing_calc_ft' in s.columns else np.zeros(len(s))
    east_arr = s['Easting_calc_ft'].to_numpy(float) if 'Easting_calc_ft' in s.columns else np.zeros(len(s))

    def interp_at_md(md, arr):
        if pd.isna(md):
            return np.nan
        md = float(md)
        if md < md_arr.min() or md > md_arr.max():
            return np.nan
        return float(np.interp(md, md_arr, arr))

    def segment_between(md1, md2):
        if pd.isna(md1) or pd.isna(md2):
            return np.array([]), np.array([]), np.array([])
        lo, hi = sorted([float(md1), float(md2)])
        md_points = [lo]
        md_points += s[(s['MD_ft'] > lo) & (s['MD_ft'] < hi)]['MD_ft'].tolist()
        md_points += [hi]
        md_points = np.array(sorted(set(float(x) for x in md_points)))
        ex = np.interp(md_points, md_arr, east_arr)
        ny = np.interp(md_points, md_arr, north_arr)
        tz = np.interp(md_points, md_arr, tvd_arr)
        return ex, ny, tz

    # Main survey trajectory
    if show_trajectory:
        ax.plot(east_arr, north_arr, tvd_arr, linewidth=2.0, alpha=0.55)

    # Casing strings and shoe information
    csort = shoes.sort_values('Shoe_MD_ft').reset_index(drop=True).copy()
    if not csort.empty:
        widths = np.linspace(8.0, 3.0, len(csort))
        for i, (_, row) in enumerate(csort.iterrows()):
            top_md = row.get('Casing_Top_MD_ft', np.nan)
            shoe_md = row.get('Shoe_MD_ft', np.nan)

            if show_casing:
                ex, ny, tz = segment_between(top_md, shoe_md)
                if len(ex) >= 2:
                    ax.plot(ex, ny, tz, linewidth=float(widths[i]), alpha=0.75)

            sx = interp_at_md(shoe_md, east_arr)
            sy = interp_at_md(shoe_md, north_arr)
            sz = interp_at_md(shoe_md, tvd_arr)
            if pd.notna(sx) and pd.notna(sy) and pd.notna(sz):
                if show_shoes:
                    ax.scatter([sx], [sy], [sz], s=70, marker='v', edgecolor='black', zorder=8)

                if show_shoe_labels:
                    shoe_tvd = row.get('Shoe_TVD_ft', np.nan)
                    inc = row.get('Inc_at_Shoe_deg', np.nan)
                    label = str(row.get('String_Name', 'String'))
                    if pd.notna(shoe_md) and pd.notna(shoe_tvd):
                        label += f'\nShoe {shoe_md:,.0f} MD / {shoe_tvd:,.0f} TVD'
                    elif pd.notna(shoe_md):
                        label += f'\nShoe {shoe_md:,.0f} MD'
                    if pd.notna(inc):
                        label += f'\nInc {inc:.1f}°'
                    ax.text(sx, sy, sz, '  ' + label, fontsize=7)

    # Geological tops
    if show_geology and geocalc is not None and not geocalc.empty:
        g = geocalc.dropna(subset=['MD_ft', 'TVD_ft']).sort_values('MD_ft')
        for _, row in g.iterrows():
            md = float(row['MD_ft'])
            gx = interp_at_md(md, east_arr)
            gy = interp_at_md(md, north_arr)
            gz = interp_at_md(md, tvd_arr)
            if pd.notna(gx) and pd.notna(gy) and pd.notna(gz):
                ax.scatter([gx], [gy], [gz], s=35, marker='s')
                ax.text(gx, gy, gz, f"  {row['Formation_Name']}\n  MD {md:,.0f}", fontsize=7)

    # Surface and TD markers
    if show_surface_td:
        ax.scatter([east_arr[0]], [north_arr[0]], [tvd_arr[0]], s=50, marker='o')
        ax.text(east_arr[0], north_arr[0], tvd_arr[0], '  Surface', fontsize=8)
        ax.scatter([east_arr[-1]], [north_arr[-1]], [tvd_arr[-1]], s=50, marker='o')
        ax.text(east_arr[-1], north_arr[-1], tvd_arr[-1], '  TD', fontsize=8)

    well_name = str(well.get('Well_Name', 'Deepwater Well'))
    ax.set_title(f'{well_name} - 3D Directional WBS', fontsize=15, fontweight='bold', pad=20)

    ax.set_xlabel('Easting (ft)', labelpad=10)
    ax.set_ylabel('Northing (ft)', labelpad=10)
    ax.set_zlabel('TVD (ft)', labelpad=10)
    ax.invert_zaxis()
    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    # Header coordinate information
    if show_coordinate_box:
        surf_x = well.get('Surface_X', '')
        surf_y = well.get('Surface_Y', '')
        bh_x = well.get('BottomHole_X', '')
        bh_y = well.get('BottomHole_Y', '')
        header_lines = []
        if surf_x != '' or surf_y != '':
            header_lines.append(f'Surface X={surf_x}, Y={surf_y}')
        if bh_x != '' or bh_y != '':
            header_lines.append(f'Bottom Hole X={bh_x}, Y={bh_y}')
        if header_lines:
            ax.text2D(0.98, 0.98, '\n'.join(header_lines), transform=ax.transAxes,
                      ha='right', va='top', fontsize=8.5,
                      bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='0.55'))

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

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Daily Operations",
        "Well / Casing",
        "Directional Survey",
        "Geological Tops",
        "MD ↔ TVD Calculator",
        "Wellbore Schematic",
        "Directional WBS",
        "3D Directional WBS",
    ]
)

with tab0:
    st.subheader("Daily Operations")

    st.write(
        "Upload the latest Daily Drilling Report and latest directional survey."
    )

    latest_ddr = st.file_uploader(
        "Upload Latest DDR",
        type=["pdf", "docx", "xlsx", "xls"],
        key="daily_ops_latest_ddr"
    )

    latest_directional_survey = st.file_uploader(
        "Upload Latest Directional Survey",
        type=["xlsx", "xls", "csv"],
        key="daily_ops_latest_directional_survey"
    )

    if latest_ddr is not None:
        st.success(f"DDR uploaded: {latest_ddr.name}")

    if latest_directional_survey is not None:
        st.success(
            f"Directional survey uploaded: {latest_directional_survey.name}"
        )

    run_daily_update = st.button(
        "Run Daily Update",
        type="primary",
        key="daily_ops_run_daily_update"
    )

    if run_daily_update:

        if latest_ddr is None and latest_directional_survey is None:
            st.warning(
                "Please upload both the Latest DDR and Latest Directional Survey."
            )

        elif latest_ddr is None:
            st.warning(
                "Please upload the Latest DDR before running the daily update."
            )

        elif latest_directional_survey is None:
            st.warning(
                "Please upload the Latest Directional Survey before running the daily update."
            )

        else:
            st.success("Both files are ready for processing.")
                try:
        # Read uploaded directional survey
        if latest_directional_survey.name.lower().endswith(".csv"):
            daily_survey_df = pd.read_csv(
                latest_directional_survey
            )
        else:
            daily_survey_df = pd.read_excel(
                latest_directional_survey
            )

        st.subheader(
            "Latest Directional Survey Preview"
        )

        st.dataframe(
            daily_survey_df,
            use_container_width=True
        )

    except Exception as e:
        st.error(
            f"Could not read the directional survey: {e}"
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

with tab6:
    fig_dir = draw_directional_wbs(well, survey, shoes, geocalc)
    st.pyplot(fig_dir, use_container_width=False)

    dir_png_buf = io.BytesIO()
    fig_dir.savefig(dir_png_buf, format="png", dpi=220, bbox_inches="tight")
    dir_png_buf.seek(0)

    dir_pdf_buf = io.BytesIO()
    fig_dir.savefig(dir_pdf_buf, format="pdf", bbox_inches="tight")
    dir_pdf_buf.seek(0)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Directional WBS PNG",
            data=dir_png_buf.getvalue(),
            file_name="directional_wbs.png",
            mime="image/png",
        )
    with d2:
        st.download_button(
            "Download Directional WBS PDF",
            data=dir_pdf_buf.getvalue(),
            file_name="directional_wbs.pdf",
            mime="application/pdf",
        )

with tab7:
    st.subheader("3D display controls")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        show_trajectory = st.toggle("Trajectory", value=True, key="3d_trajectory")
        show_casing = st.toggle("Casing strings", value=True, key="3d_casing")
    with t2:
        show_shoes = st.toggle("Shoe markers", value=True, key="3d_shoes")
        show_shoe_labels = st.toggle("Shoe labels", value=True, key="3d_shoe_labels")
    with t3:
        show_geology = st.toggle("Geological tops", value=True, key="3d_geology")
        show_surface_td = st.toggle("Surface / TD", value=True, key="3d_surface_td")
    with t4:
        show_coordinate_box = st.toggle("Coordinate box", value=True, key="3d_coord_box")

    a1, a2 = st.columns(2)
    with a1:
        view_elev = st.slider("3D elevation angle", min_value=5, max_value=80, value=24, step=1, key="3d_elev")
    with a2:
        view_azim = st.slider("3D azimuth angle", min_value=-180, max_value=180, value=-58, step=1, key="3d_azim")

    fig_3d = draw_3d_directional_wbs(
        well,
        survey,
        shoes,
        geocalc,
        show_trajectory=show_trajectory,
        show_casing=show_casing,
        show_shoes=show_shoes,
        show_shoe_labels=show_shoe_labels,
        show_geology=show_geology,
        show_surface_td=show_surface_td,
        show_coordinate_box=show_coordinate_box,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    st.pyplot(fig_3d, use_container_width=False)

    w3_png = io.BytesIO()
    fig_3d.savefig(w3_png, format="png", dpi=220, bbox_inches="tight")
    w3_png.seek(0)

    w3_pdf = io.BytesIO()
    fig_3d.savefig(w3_pdf, format="pdf", bbox_inches="tight")
    w3_pdf.seek(0)

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download 3D WBS PNG",
            data=w3_png.getvalue(),
            file_name="3d_directional_wbs.png",
            mime="image/png",
        )
    with e2:
        st.download_button(
            "Download 3D WBS PDF",
            data=w3_pdf.getvalue(),
            file_name="3d_directional_wbs.pdf",
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

