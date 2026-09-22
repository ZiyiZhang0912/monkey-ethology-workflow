# Architecture

Pipeline stages exposed by `EthologyAgent`:

1. **Ingest** — DeepLabCut `.h5` / multi-header `.csv` or flat `Joint_{x,y,p}` tables
2. **Preprocess** — occlusion interpolation, position anomaly repair, optional median smoothing
3. **Features** — sliding-window kinematics with adaptive center of mass
4. **Classify** — rule-based locomotion (Static / Walking / Climbing); optional stereotypy and circling plugins
5. **Visualize** — ethogram, time budget, trajectory / heatmap style figures

Configuration is YAML-driven (`configs/default.yaml` plus optional cage / bodyparts / animal overlays). Animal IDs are session labels; per-animal YAML is optional.
