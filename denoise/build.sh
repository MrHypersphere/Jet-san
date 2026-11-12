#!/usr/bin/env bash
# Builds the jetdenoise Python module in this folder.
# Needs: a C++17 compiler and `pip install pybind11`.
set -euo pipefail
cd "$(dirname "$0")"
c++ -O3 -march=native -std=c++17 -shared -fPIC -Wall -Wextra \
    $(python3 -m pybind11 --includes) bindings.cpp \
    -o jetdenoise$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "built: $(ls jetdenoise*.so)"
