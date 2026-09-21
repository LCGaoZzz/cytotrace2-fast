#!/usr/bin/env python3
"""Export an explicitly selected h5ad expression slot to native cytotrace2 TSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import signal

from run_cytotrace2 import file_info, identifiers, interrupted, require


def export_matrix(matrix, genes, cells, output, metadata, gene_chunk_size=16):
    """Densify only one gene block at a time; never normalize or rename IDs."""
    import numpy as np

    identifiers(genes, "gene")
    identifiers(cells, "cell")
    require(matrix.shape == (len(cells), len(genes)), "matrix/ID dimensions differ")
    require(type(gene_chunk_size) is int and gene_chunk_size >= 1, "gene_chunk_size must be positive")
    output = Path(output)
    provenance = output.with_name(output.name + ".provenance.json")
    require(not output.exists() and not output.is_symlink(), "output already exists")
    require(not provenance.exists() and not provenance.is_symlink(), "provenance already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            created.append(output)
            stream.write("\t".join(cells) + "\n")  # NO gene-name header or leading tab.
            maximum = 0.0
            for start in range(0, len(genes), gene_chunk_size):
                block = matrix[:, start:start + gene_chunk_size]
                if hasattr(block, "toarray"):
                    block = block.toarray()
                block = np.asarray(block)
                require(np.isfinite(block).all() and (block >= 0).all(),
                        "expression must be finite and nonnegative; log/scale provenance remains the caller's responsibility")
                maximum = max(maximum, float(block.max()))
                for offset in range(block.shape[1]):
                    stream.write(genes[start + offset] + "\t" +
                                 "\t".join(format(float(v), ".17g") for v in block[:, offset]) + "\n")
            require(maximum > 0, "all-zero expression matrix")
        report = {**metadata, "cells": len(cells), "genes": len(genes),
                  "format": "native_tsv_cells_only_header", "transformations": [],
                  "gene_chunk_size": gene_chunk_size, "output": file_info(output)}
        with provenance.open("x", encoding="utf-8") as stream:
            created.append(provenance)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return report
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--matrix-source", required=True, help="X, raw.X or layers/NAME; never inferred")
    parser.add_argument("--gene-column", default="var_names", help="var_names or a column in the selected slot's var")
    parser.add_argument("--expression-scale", choices=["counts", "CPM", "TPM"], required=True)
    parser.add_argument("--gene-chunk-size", type=int, default=16)
    opts = parser.parse_args(argv)
    data = None
    try:
        import anndata

        source = Path(opts.input).expanduser().resolve(strict=True)
        output = Path(opts.output).expanduser().absolute()
        require(source != output.resolve(), "input and output must differ")
        data = anndata.read_h5ad(source, backed="r")
        if opts.matrix_source == "X":
            matrix, var = data.X, data.var
        elif opts.matrix_source == "raw.X":
            require(data.raw is not None, "raw.X is absent")
            matrix, var = data.raw.X, data.raw.var
        elif opts.matrix_source.startswith("layers/"):
            key = opts.matrix_source[len("layers/"):]
            require(key in data.layers, f"missing layer: {key}")
            matrix, var = data.layers[key], data.var
        else:
            raise ValueError("matrix-source must be X, raw.X or layers/NAME")
        require(matrix is not None, "selected expression matrix is absent")
        if opts.gene_column == "var_names":
            genes = list(map(str, var.index))
        else:
            require(opts.gene_column in var.columns, f"missing gene column: {opts.gene_column}")
            require(not var[opts.gene_column].isna().any(), "gene column contains missing IDs")
            genes = list(map(str, var[opts.gene_column]))
        report = export_matrix(matrix, genes, list(map(str, data.obs_names)), output,
                               {"input": file_info(source), "matrix_source": opts.matrix_source,
                                "gene_column": opts.gene_column, "expression_scale": opts.expression_scale,
                                "anndata_version": anndata.__version__}, opts.gene_chunk_size)
        print(json.dumps(report, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except ImportError as error:
        print(f"Use a prepared interpreter with anndata/numpy/scipy: {error}", file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if data is not None:
            data.file.close()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
