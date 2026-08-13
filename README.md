
# Deepwater Wellbore Schematic Builder — Prototype

This Streamlit prototype reads the companion Excel input template and automatically:

- Uses the directional survey as the master MD ↔ TVD conversion source.
- Calculates TVD from MD by interpolation.
- Finds MD solution(s) for a specified TVD.
- Calculates minimum-curvature TVD if the survey does not contain a complete TVD column.
- Calculates inclination and azimuth at each casing shoe.
- Calculates maximum inclination reached at/before each casing shoe.
- Interpolates temperature at each casing shoe from the TVD-temperature profile.
- Calculates MD/TVD/TVDSS/inclination/temperature for geological tops.
- Links FIT/LOT results to the applicable casing string.
- Draws an initial auto-generated wellbore schematic.
- Exports calculated tables to Excel.

## Run locally

1. Install Python 3.11+.
2. Open a terminal in this folder.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the app:

   streamlit run app.py

5. Upload `deepwater_wbs_input_template.xlsx`.

## Engineering conventions

- MD, TVD and TVDSS are stored separately.
- Current prototype assumes RKB is positive above MSL and calculates TVDSS as `TVD - RKB`.
- Temperature interpolation is based on TVD.
- For TVD → MD, the program searches every survey segment. If a trajectory gives multiple MD values at the same TVD, all solutions are flagged.
- Minimum-curvature calculations are used when TVD is not fully supplied in the directional survey.

## Recommended next development phase

1. Add a professional WBS renderer that matches the supplied engineering schematic format.
2. Add editable UI forms rather than Excel-only inputs.
3. Add mud program, cement system, pore pressure/fracture gradient and BHA columns.
4. Add revision control and multiple-well database storage.
5. Add PDF/PNG export.


## Stage 2 additions

- Three-column professional WBS layout:
  - geological tops on the left
  - nested casing/cement in the center
  - shoe/FIT/LOT/temperature/inclination callouts on the right
- TOC markers for each string.
- TD marker and mudline reference.
- Direct PNG and PDF schematic export from the Streamlit app.
