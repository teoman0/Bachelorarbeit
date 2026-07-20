# DINOv3-Region-Head-Plan

Stand: 2026-07-20

## Ziel

Die bisherige DINOv3-Regionenauswertung nutzte das global trainierte
DINOv3-Partial-Fine-Tuning-Modell direkt auf CVAT-Bounding-Box-Regionen. Auf
den lokalen Regionen traten deutliche Verwechslungen auf. Als naechster
methodischer Schritt wird deshalb ein kleiner lokaler Klassifikationskopf auf
DINOv3-Features der CVAT-Region-Crops vorbereitet.

Der Ansatz bleibt eine rechteckbasierte lokale Regionenanalyse und ist keine
semantische Segmentierung.

## Eingaben

Regionentabelle:

```text
outputs/region_analysis/cvat_region_analysis/region_annotations.csv
```

DINOv3-Partial-Fine-Tuning-Checkpoint:

```text
outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/checkpoints/best_model.pt
```

Die lokalen Originalbilder werden ueber das `manual_all`-Verzeichnis geladen.
Der lokale Pfad wird per `BMW25_MANUAL_ALL_ROOT` oder `--manual-root`
uebergeben und nicht versioniert.

## Split-Regel

Es werden keine neuen Splits erzeugt. Der Workflow verwendet ausschliesslich
die vorhandene Regionentabelle, die aus dem bestehenden grouped split manifest
abgeleitet wurde.

- Training: nur `split=train`
- Checkpoint-Auswahl und Evaluation: nur `split=val`
- Test: vollstaendig ausgeschlossen

`Nicht_bewertbar` wird standardmaessig ausgeschlossen und geht nicht in die
4-Klassen-Metrik ein.

## Modell

Das Backbone ist `facebook/dinov3-vitb16-pretrain-lvd1689m` mit den Gewichten
aus dem bestehenden DINOv3-Partial-Fine-Tuning-Checkpoint. Es bleibt
eingefroren. Trainiert wird nur ein kleiner Head:

```text
Linear 768 -> 128
ReLU
Dropout 0.2
Linear 128 -> 4
```

Der Start-Crop-Modus ist `stretch_resize`, weil diese Variante in der
Validierungsablation der direkten Regioneninferenz am besten abschnitt.
`pad_square` bleibt als dokumentierte Option erhalten.

## Training

| Parameter | Wert |
| --- | --- |
| Experiment | `dinov3_region_head_bmw25_seed42` |
| Backbone | frozen DINOv3 |
| Head | MLP |
| Feature-Dimension | 768 |
| Klassen | 4 |
| Crop-Mode | `stretch_resize` |
| Kontext-Margin | `0.0` |
| Batch Size | 16 |
| Epochen | 100 |
| Patience | 15 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Checkpoint-Metrik | Validation Macro-F1 |
| Seed | 42 |

Class Weights werden aus der Train-Regionenverteilung berechnet, sofern die
aktive Umgebung dies unterstuetzt.

## Augmentierung

Nur fuer Train-Regionen sind leichte Augmentierungen vorgesehen:

- leichte Helligkeitsvariation;
- leichte Kontrastvariation;
- horizontale und vertikale Flips.

Da Schleifspuren richtungsabhaengig sein koennen, muessen Flips methodisch
transparent eingeordnet werden. Validation bleibt deterministisch.

## Lokale Befehle

Dry-Run ohne Modellladen:

```powershell
python scripts/train_dinov3_region_head.py `
  --dry-run
```

Modellcheck ohne Training:

```powershell
python scripts/train_dinov3_region_head.py `
  --check-model
```

Smoke-Test ohne Checkpoints:

```powershell
python scripts/train_dinov3_region_head.py `
  --manual-root <lokales_manual_all_verzeichnis> `
  --smoke-test
```

Spaeterer echter Lauf nur nach separater Bestaetigung:

```powershell
python scripts/train_dinov3_region_head.py `
  --manual-root <lokales_manual_all_verzeichnis> `
  --allow-training `
  --crop-mode stretch_resize `
  --context-margin 0.0
```

## Lokale Artefakte

Bei `--allow-training` werden lokale Artefakte geschrieben nach:

```text
outputs/region_analysis/dinov3_region_head_bmw25_seed42/
```

Moegliche Dateien:

- `run_metadata.json`
- `training_log.csv`
- `predictions_val.csv`
- `val_metrics.json`
- `val_metrics.csv`
- `confusion_matrix_val.csv`
- `checkpoints/best_region_head.pt`
- `checkpoints/last_region_head.pt`

Diese Outputs, Predictions, Checkpoints, Gewichte und lokale Artefakte bleiben
ignored und werden nicht committed.
