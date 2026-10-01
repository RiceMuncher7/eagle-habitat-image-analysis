Eagle Habitat Image Analysis Project

This folder contains the final outputs for a computer vision project comparing two methods for estimating environmental composition in eagle habitat photographs.

Main report:
- Eagle_Habitat_Image_Analysis_Report.pdf

Main data outputs:
- comparison_summary.csv: overall Classical vs DL comparison statistics
- year_means_compare.csv: year-level water and vegetation averages
- merged_per_image.csv: per-image Classical vs DL values
- audit_top_disagreements.csv: largest disagreement cases
- audit_notes.txt: qualitative error analysis notes

Figures:
- figures/trend_water_frac.png
- figures/trend_vegetation_frac.png
- figures/scatter_water_frac.png
- figures/scatter_vegetation_frac.png
- figures/hist_water_frac.png
- figures/hist_vegetation_frac.png

Project framing:
This project compares classical color-threshold segmentation and deep semantic segmentation for estimating water and vegetation coverage in eagle habitat photographs. The main finding is that the two methods produce systematically different measurements under real-world photographic conditions such as reflections, winter vegetation, sky-heavy framing, snow/ice, and shoreline ambiguity.