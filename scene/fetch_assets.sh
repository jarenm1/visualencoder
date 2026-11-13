#!/usr/bin/env bash
# Fetch CC0 assets: Kenney kits (geometry, used by scene/build_city.py) into assets/kenney/<kit>/
# and Poly Haven surface textures (used by the renderer) into assets/polyhaven/.
# Mirrors on OpenGameArt; kenney.nl itself gates downloads behind a JS dialog.
set -euo pipefail
dest="${1:-assets/kenney}"
base="https://opengameart.org/sites/default/files"
declare -A kits=(
  [city-commercial]="kenney_city-kit-commercial_2.1.zip"
  [city-suburban]="kenney_city-kit-suburban_2.0.zip"
  [city-roads]="kenney_city-kit-roads_2.2.zip"
  [nature]="Nature%20Kit%20%282.1%29.zip"
  [cars]="kenney_car-kit_3.1.zip"
)
mkdir -p "$dest"
for kit in "${!kits[@]}"; do
  if [[ -d "$dest/$kit" ]]; then continue; fi
  tmp=$(mktemp --suffix=.zip)
  curl -fsSL -A "Mozilla/5.0" "$base/${kits[$kit]}" -o "$tmp"
  unzip -q "$tmp" -d "$dest/$kit"
  rm "$tmp"
  grep -q "CC0" "$dest/$kit/License.txt" || { echo "$kit: licence is not CC0" >&2; exit 1; }
  echo "fetched $kit"
done
python3 "$(dirname "$0")/fetch_polyhaven.py" assets/polyhaven
