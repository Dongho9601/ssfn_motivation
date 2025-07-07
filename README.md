# ssfn_motivation

This repo provides simple utilities to measure the mean squared error (MSE)
introduced by various numeric formats. Available quantization methods are:

* **FP16**  – IEEE half precision
* **BFP16** – bfloat16
* **MXINT8** – block-wise int8 with dynamic scaling
* **MXFP8**  – block-wise 8-bit floating point
* **MXFP6**  – block-wise 6-bit floating point
* **MXFP4**  – block-wise 4-bit floating point

Implementation details for the MX formats are documented in the PDF
specifications under `docs/`.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```
