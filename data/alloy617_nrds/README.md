# Alloy 617 NRDS Compact Subset

This directory contains a compact, native-data subset from the INL NRDS
release **Focused Ion Beam Tomography of Alloy 617 Corroded in Molten
Chloride Salt**.

## Specimen And Provenance

- Material: Alloy 617 / Inconel 617.
- Exposure: eutectic sodium chloride and magnesium chloride molten salt at
  700 C for 1000 hours.
- Instrument: FEI G4 Helios Hydra Plasma-FIB.
- Modalities: SEM image, EDS elemental image, and EBSD supporting data.
- Tomography slice offset: 100 nm.
- DOI: https://doi.org/10.48806/2287679
- License: Creative Commons Attribution (CC BY 4.0).

## Downloaded Subset

The downloaded files use the common tomography label `006` where an
individual-slice resource was exposed through the public NRDS pages:

| Local file | Role | Native shape / size |
| --- | --- | --- |
| `eds/slice_006/dat/EBSD-SliceImage-006_0_*.dat` | Numeric EDS maps for Al, Cl, Co, Cr, F, Fe, Mg, Mn, Mo, Na, Ni, O, Si, Ti, and W | 15 headered count maps; each payload is 549 x 478 |
| `eds/slice_006/png/EBSD-SliceImage-006_*.png` | Official rendered previews for the same 15 EDS maps | 1536 x 1157 RGBA PNG |
| `eds/slice_006/Cr_EDS_006.tif` | Earlier chromium-only rendered map retained for the first replay regression | 1536 x 1157 grayscale TIFF |
| `sem/slice_006/SEM_Image_SliceImage_006.tif` | Matching SEM morphology context | 1536 x 1024 grayscale TIFF |
| `ebsd/EBSD_SliceImage_006.dat` | Matching EBSD slice data | 26.8 MB binary data file |
| `ebsd/A617_Test_6-7_EBSD.json` | EBSD reconstruction/export metadata | 20.5 KB JSON |

The 15 numeric EDS maps have a common `549 x 478` native grid. Their first two
unsigned 32-bit values declare columns and rows; the next `549 * 478` values
are the map payload. The payloads reproduce the corresponding PNG patterns
(for example, Cr correlation is approximately `0.98` after visualization
resampling). The EBSD metadata gives native spacing of `100.0 nm` in X and
`86.6 nm` in Y. The SEM and EDS renderings are not pixel-aligned and require
registered multimodal handling.

## BALANCE-NM Use

This subset is now a runnable multi-element BALANCE-NM replay source through
`configs/alloy617_multielement_replay.yaml`. The replay uses the numeric DAT
payloads, not the color previews, with multi-objective gradient, anomaly,
clustering, and inclusion interest maps. Because response-factor calibration
and acquisition dwell metadata are not supplied with the selected maps, the
channels are treated as `uncalibrated_counts`: the analysis supports spatial
pattern targeting and withheld-map reconstruction tests, not quantitative
composition or corrosion-chemistry conclusions.

The all-element slice-006 EDS DAT and PNG files total approximately `26 MB`.
The full approximately `33 GB` public tomography release has not been
downloaded wholesale.

## Source Pages

- https://nrds.inl.gov/dataset/a617_test6-7_images_ebsd
- https://nrds.inl.gov/dataset/a617_test6-7_images_ebsd___eds
- https://nrds.inl.gov/dataset/a617_test6-7_images_sem_image
