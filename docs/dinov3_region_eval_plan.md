# DINOv3-Regionenauswertung-Plan

Stand: 2026-07-20

## Ziel

Diese Notiz beschreibt eine lokale rechteckbasierte Regionenauswertung fuer
die manuell annotierten CVAT-Bounding-Boxes. Das beste
DINOv3-Partial-Fine-Tuning-Modell wird dabei nur fuer Inferenz verwendet. Es
wird kein weiteres Training gestartet.

Die Analyse prueft, ob das global trainierte Modell lokale Bildbereiche
plausibel klassifiziert. Sie ist keine echte semantische Segmentierung, weil
die Annotationen rechteckige Regionen und keine pixelgenauen Masken sind.

## Eingaben

Regionentabelle:

```text
outputs/region_analysis/cvat_region_analysis/region_annotations.csv
```

DINOv3-Checkpoint:

```text
outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/checkpoints/best_model.pt
```

Die lokalen Originalbilder werden fuer `--allow-evaluate` ueber das
`manual_all`-Verzeichnis bereitgestellt. Der lokale Pfad wird nicht
versioniert, sondern per `BMW25_MANUAL_ALL_ROOT` oder `--manual-root`
uebergeben.

## Split-Regel

Standardmaessig wird nur `split=val` ausgewertet. Testregionen sind
ausgeschlossen und duerfen nicht fuer Modellwahl, Schwellenwerte oder
qualitative Validierungsentscheidungen verwendet werden. Es werden keine neuen
Splits erzeugt; die Regionentabelle basiert auf dem bestehenden grouped split
manifest.

## Umgang mit `Nicht_bewertbar`

Regionen mit `mapped_label = Nicht_bewertbar` sind keine der vier globalen
Schleifgradklassen. Sie koennen mit `--include-nicht-bewertbar` separat
inferiert und in der Prediction-Tabelle berichtet werden. In Accuracy,
Balanced Accuracy, Macro-F1, klassenweisen Metriken und Confusion Matrix
gehen sie nicht ein.

## Crop-Verarbeitung

Jede Bounding Box wird vor dem Ausschneiden an die Bildgrenzen geclippt. Als
Standard wird `crop_mode=pad_square` verwendet:

1. Rechteck aus dem Originalbild ausschneiden.
2. Crop quadratisch auffuellen.
3. Quadratischen Crop auf `224 x 224` skalieren.

Optional ist `stretch_resize` vorgesehen, um den Crop direkt auf `224 x 224`
zu skalieren. Dieser Modus ist nur fuer einen spaeteren methodischen Vergleich
gedacht und nicht der Standard.

## Lokale Befehle

Dry-Run ohne Modellladen und ohne Dateiausgabe:

```powershell
python scripts/evaluate_dinov3_regions.py `
  --dry-run `
  --split val
```

Modell- und Checkpoint-Pruefung ohne Auswertung:

```powershell
python scripts/evaluate_dinov3_regions.py `
  --check-model `
  --split val
```

Lokale Val-Auswertung nach separater Bestaetigung:

```powershell
python scripts/evaluate_dinov3_regions.py `
  --allow-evaluate `
  --manual-root <lokales_manual_all_verzeichnis> `
  --split val `
  --crop-mode pad_square `
  --include-nicht-bewertbar
```

## Lokale Artefakte

Bei `--allow-evaluate` werden nur lokale, ignorierte Artefakte geschrieben:

```text
outputs/region_analysis/dinov3_region_eval_bmw25_seed42/
```

Geplante Dateien:

- `predictions_regions_val.csv`
- `val_region_metrics.json`
- `val_region_metrics.csv`
- `confusion_matrix_val_regions.csv`

Crops oder Overlays werden standardmaessig nicht exportiert. Outputs,
Predictions, Crops, Checkpoints und Gewichte bleiben lokal und werden nicht
committed.
