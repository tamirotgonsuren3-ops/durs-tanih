#!/usr/bin/env bash
# Run the Face Recognition Attendance Streamlit app.
# dlib (installed from conda package cache) needs these libs on the search path.
export LD_LIBRARY_PATH="/tmp:/home/bumblebee/miniconda3/pkgs/libcblas-3.11.0-11_h0358290_openblas/lib:/home/bumblebee/miniconda3/lib:${LD_LIBRARY_PATH:-}"
cd "$(dirname "$0")"
exec streamlit run app.py "$@"
