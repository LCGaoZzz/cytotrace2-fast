#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-export the bundled model assets in assets/ from the official cytotrace2-py package.

Reads the pretrained resources shipped inside an installed official `cytotrace2-py`
package (torch .pt models, background graph, feature list, gene mapping tables) and
writes the .npz / text assets this repository's Rust binary consumes, plus the
assets/MANIFEST.json provenance manifest.

Run this inside an environment that has the official package installed (or at least
its resources directory) together with `torch`, `pandas` and `numpy` — e.g. the
official cytotrace2 environment itself:

    python scripts/export_assets.py \
        --resources /path/to/site-packages/cytotrace2_py/resources \
        --out assets

Use --verify (with --ref, default: the --out directory) to re-derive everything in
memory and byte-compare against an existing assets directory without writing.

Notes on exact reproduction
---------------------------
* Model npz files are written with numpy.savez (uncompressed, ZIP_STORED) in
  state-dict key order; entry timestamps default to 1980-01-01, so output is
  byte-reproducible.
* background.pt is an uncoalesced torch.sparse_coo tensor; it is coalesced before
  export so indices/values are in canonical sorted order.
* map_human_ortholog.tsv reproduces the official build_mapping_dict() logic
  (pandas group-by double top-hit selection + feature self-mapping). The group-by /
  sort_values tie-breaking is pandas-version-sensitive: this script reproduced the
  bundled assets byte-identically under pandas 2.3.3 / numpy 1.26.4 / torch 2.14.0.
"""

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
import pandas as pd
import torch

MODEL_N = 19


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_label(resources, path):
    """Return portable provenance instead of leaking a local absolute path."""
    rel = os.path.relpath(path, resources).replace(os.sep, "/")
    return "cytotrace2_py/resources/" + rel


# ---- official selection logic (cytotrace2_py/common/gen_utils.py) ----

def top_hit_human(x):
    r = x.sort_values("%id. target Mouse gene identical to query gene")
    return r.iloc[-1]


def top_hit_mouse(x):
    r = x.sort_values("%id. query gene identical to target Mouse gene")
    return r.iloc[-1]


def build_human_ortholog(resources):
    human_mapping = pd.read_csv(os.path.join(resources, "mart_export.txt"), sep="\t").dropna().reset_index()
    mapping_unique = human_mapping.groupby("Gene name").apply(top_hit_human)
    mapping_unique = mapping_unique.groupby("Mouse gene name").apply(top_hit_mouse)
    mt_dict = dict(zip(mapping_unique["Gene name"].values, mapping_unique["Mouse gene name"].values))
    features = pd.read_csv(os.path.join(resources, "features_model_training.csv"))["0"]
    for gene in features[~features.isin(mt_dict.values())].values:
        mt_dict[gene.upper()] = gene
    return mt_dict


def dump_tsv(path, d):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join("%s\t%s\n" % (k, d[k]) for k in sorted(d)))


def export(resources, out):
    os.makedirs(out, exist_ok=True)
    files = {}

    # ---- 19 ensemble models: state dict -> npz (key order preserved) ----
    # NOTE: the manifest walks the models in lexicographic filename order
    # (model1, model10, model11, ..., model19, model2, ..., model9); this is
    # also the ensemble evaluation order recorded as MANIFEST "model_order".
    model_order = sorted("model%d.pt" % i for i in range(1, MODEL_N + 1))
    for name in model_order:
        i = int(name[len("model"):-len(".pt")])
        src = os.path.join(resources, "models", name)
        sd = torch.load(src, map_location="cpu", weights_only=False)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        arrays = {k: v.detach().cpu().numpy() for k, v in sd.items()}
        dst = os.path.join(out, "model%d.npz" % i)
        np.savez(dst, **arrays)
        files["model%d.npz" % i] = {
            "sha256": sha256_file(dst),
            "bytes": os.path.getsize(dst),
            "kind": "model_state_dict",
            "source": source_label(resources, src),
            "keys": {k: {"shape": list(v.shape), "dtype": v.dtype.name} for k, v in arrays.items()},
        }

    # ---- background graph: sparse COO -> npz ----
    src = os.path.join(resources, "background.pt")
    coo = torch.load(src, map_location="cpu", weights_only=False).coalesce()
    dst = os.path.join(out, "background.npz")
    np.savez(dst,
             indices=coo.indices().numpy(),
             values=coo.values().numpy(),
             shape=np.asarray(coo.shape, dtype=np.int64))
    files["background.npz"] = {
        "sha256": sha256_file(dst),
        "bytes": os.path.getsize(dst),
        "kind": "background_coo",
        "source": source_label(resources, src),
        "indices_shape": list(coo.indices().numpy().shape),
        "indices_dtype": str(coo.indices().numpy().dtype),
        "values_dtype": str(coo.values().numpy().dtype),
        "dense_shape": list(coo.shape),
        "nnz": int(coo._nnz()),
    }

    # ---- feature list ----
    src = os.path.join(resources, "features_model_training.csv")
    features = pd.read_csv(src)["0"]
    dst = os.path.join(out, "features.txt")
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(features) + "\n")
    files["features.txt"] = {
        "sha256": sha256_file(dst),
        "bytes": os.path.getsize(dst),
        "kind": "feature_list",
        "n": int(len(features)),
        "source": source_label(resources, src),
    }

    # ---- gene mapping tables ----
    ortho = build_human_ortholog(resources)
    dst = os.path.join(out, "map_human_ortholog.tsv")
    dump_tsv(dst, ortho)
    files["map_human_ortholog.tsv"] = {
        "sha256": sha256_file(dst),
        "bytes": os.path.getsize(dst),
        "n": len(ortho),
        "kind": "human_gene->mouse_gene",
    }

    halias = pd.read_csv(os.path.join(resources, "human_alias_list.txt"), sep="\t")
    d = dict(zip(halias["Alias or Previous Gene name"].values, halias["Mouse gene name"].values))
    dst = os.path.join(out, "map_human_alias.tsv")
    dump_tsv(dst, d)
    files["map_human_alias.tsv"] = {
        "sha256": sha256_file(dst),
        "bytes": os.path.getsize(dst),
        "n": len(d),
        "kind": "human_alias->mouse_gene",
    }

    malias = pd.read_csv(os.path.join(resources, "mouse_alias_list.txt"), sep="\t")
    d = dict(zip(malias["alias"].values, malias["mmgene"].values))
    dst = os.path.join(out, "map_mouse_alias.tsv")
    dump_tsv(dst, d)
    files["map_mouse_alias.tsv"] = {
        "sha256": sha256_file(dst),
        "bytes": os.path.getsize(dst),
        "n": len(d),
        "kind": "mouse_alias->mmgene",
    }

    # ---- reference constants (hex dumps of the float64 bit patterns) ----
    obj = {
        "linspace_0_1_7_hex": [struct.pack("<d", v).hex() for v in np.linspace(0, 1, 7)],
        "arange7_div6_hex": [struct.pack("<d", v).hex() for v in (np.arange(7) / 6)],
    }
    dst = os.path.join(out, "reference_constants.json")
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(obj, indent=1))
    files["reference_constants.json"] = {
        "sha256": sha256_file(dst),
        "kind": "constants",
    }

    # ---- manifest ----
    manifest = {
        "files": files,
        "model_order": model_order,
        "notes": {"walk_order_is_lex": True},
    }
    with open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(manifest, indent=1))
    return files


def find_resources():
    try:
        import pkg_resources
        return pkg_resources.resource_filename("cytotrace2_py", "resources")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Re-export cytotrace2-fast assets from the official package resources")
    ap.add_argument("--resources", default=None, help="cytotrace2_py/resources directory (default: auto-detect installed package)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"),
                    help="output assets directory (default: <repo>/assets)")
    ap.add_argument("--verify", action="store_true", help="do not write; re-derive into a temp dir and byte-compare against --ref")
    ap.add_argument("--ref", default=None, help="existing assets directory to compare against in --verify mode (default: --out)")
    args = ap.parse_args()

    resources = args.resources or find_resources()
    if not resources or not os.path.isdir(resources):
        sys.exit("resources directory not found; pass --resources /path/to/cytotrace2_py/resources")
    ref = args.ref or args.out

    if args.verify:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            files = export(resources, td)
            ok = True
            names = sorted(files) + ["MANIFEST.json"]
            for name in names:
                a = os.path.join(td, name)
                b = os.path.join(ref, name)
                if not os.path.exists(b):
                    print("MISSING in ref: %s" % name); ok = False; continue
                if sha256_file(a) != sha256_file(b):
                    print("DIFFERS: %s" % name); ok = False
                else:
                    print("ok: %s" % name)
            print("verify: %s" % ("PASS" if ok else "FAIL"))
            sys.exit(0 if ok else 1)
    else:
        files = export(resources, args.out)
        print("wrote %d files + MANIFEST.json to %s" % (len(files), args.out))


if __name__ == "__main__":
    main()
