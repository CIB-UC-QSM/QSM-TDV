#!/bin/bash

# Detener el script si ocurre algún error
set -e

# Definir la lista de valores de SNR
snr_values=(70) # 100 150)

# Iterar sobre cada valor de SNR
for snr in "${snr_values[@]}"; do
  echo "=================================================="
  echo "Iniciando entrenamiento con SNR = $snr"
  echo "=================================================="

  uv run tdv-qsm-train \
    --data /cosmos_data \
    --output-dir runs/cosmos-snr${snr}-fix \
    --epochs 10 \
    --snr "$snr" \
    --features 1 \
    --steps 5 \
    --maximum-time 0.25 \
    --learning-rate 1e-4 \
    --augmentation

  echo "Completado SNR = $snr"
  echo ""
done

echo "¡Todos los entrenamientos han finalizado con éxito!"
