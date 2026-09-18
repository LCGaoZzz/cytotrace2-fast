# NOTICE

## Upstream attribution

This repository contains **cytotrace2-fast**, a Rust reimplementation of CytoTRACE 2,
created as a derivative work of the official implementation:

- Upstream project: CytoTRACE 2 — https://github.com/digitalcytometry/cytotrace2
- Upstream authors: Minji Kang, Gunsagar S. Gulati, Erin L. Brown, Zhen Qi, Susanna Avagyan,
  Jose Juan Almagro Armenteros, Rachel Gleyzer, Wubing Zhang, Chloé B. Steen,
  Jeremy Philip D'Silva, Janella Schwab, Michael F. Clarke, Aadel A. Chaudhuri,
  Aaron M. Newman (Newman Lab, Stanford University)
- Upstream publication: *Improved reconstruction of single-cell developmental potential
  with CytoTRACE 2.* Nature Methods, 2025. doi:10.1038/s41592-025-02857-2

## Derivative relationship

- The biological method, model architecture, preprocessing design and CLI semantics
  originate from the upstream project.
- `assets/` redistributes the upstream pretrained model weights (19 ensemble models),
  the background graph, the feature list and the gene mapping tables, converted
  losslessly from the official `cytotrace2-py` package resources into numpy `.npz` /
  text form (see `scripts/export_assets.py` and `assets/MANIFEST.json`).
- The Rust source in `src/` is an independent reimplementation of the official Python
  pipeline, validated for output parity against the official implementation
- `vendor/faer` is a vendored fork of the faer linear-algebra crate (MIT license, upstream:
  https://github.com/sarah-quin/faer), pinned as the numerical backend.

## Non-commercial restriction

The upstream software is licensed by Stanford University for **non-commercial use only**
(see `LICENSE`, Stanford Non-Commercial Software License Agreement, docket S24-057).
That restriction carries over to this repository and its redistributed model assets.
Commercial entities wishing to use this software should contact Stanford University's
Office of Technology Licensing.
