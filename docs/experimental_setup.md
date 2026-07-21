# Experimenteller Aufbau

## Zweck und Geltungsbereich

Dieses Dokument ist die kanonische Beschreibung der tatsächlich ausgeführten
Experimente und ihrer aufgezeichneten Parameter. Es unterscheidet reale
Laufkonfigurationen von verworfenen Planungsständen und führt keine
Ergebniswerte des finalen Testsets.

## Gemeinsame Grundlage

Alle Experimente verwenden das versionierte gruppierte Split-Manifest
`data/splits/bmw25_grouped_split_manifest.csv`. Die globalen Modelle wurden
mit 3.225 Trainingsbildern entwickelt und auf 691 Validierungsbildern
bewertet. Der Testsplit war während der Modellentwicklung von Training,
Checkpoint-Auswahl und Leistungsbewertung ausgeschlossen.

Soweit nicht anders angegeben, wurde der Seed 42 verwendet. Trainings- und
Evaluationsartefakte liegen ausschließlich in ignorierten lokalen
Verzeichnissen. Dazu gehören Run-Metadaten, Logs, Predictions, Metrikdateien,
Confusion Matrices und Checkpoints.

## Globale Modelle

### YOLOv11n-cls

`YOLOv11n-cls` diente als vortrainierte globale Klassifikationsbaseline. Die
lokale YOLO-Dataset-Struktur bestand aus Links auf die bereits gesplitteten
Originalbilder; Bilder wurden nicht kopiert.

| Merkmal | Tatsächliche Einstellung |
| --- | --- |
| Modell | `yolo11n-cls`, vortrainiert |
| Aufgabe | globale Vier-Klassen-Bildklassifikation |
| Eingabegröße | 320 x 320 Pixel |
| Batchgröße | 16 |
| maximales Epochenbudget | 75 |
| protokollierte Epochen | 57 |
| Early Stopping | Patience 15; bester Checkpoint in Epoche 42 |
| Optimierer | Ultralytics `auto` |
| aufgezeichnetes `lr0` | 0,01 |
| aufgezeichnetes `lrf` | 0,01 |
| Weight Decay | 0,0005 |
| AMP | aktiviert |
| Device | CUDA-Gerät 0 |
| Seed | 42 |
| Config | `configs/experiments/yolov11_cls.yaml` |
| Skript | `scripts/train_yolov11_cls.py` |

Die tatsächlich aufgezeichneten Ultralytics-Argumente in `args.yaml` sind für
die Rekonstruktion des Laufs maßgeblich. Der lokale Run liegt unter
`runs/global_classification/yolov11_cls_n_imgsz320_seed42/` und enthält unter
anderem `results.csv`, `args.yaml` sowie `weights/best.pt` und
`weights/last.pt`.

### DINOv3 mit eingefrorenem Backbone und linearem Head

Für die erste DINOv3-Variante wurde der vortrainierte Backbone vollständig
eingefroren. Trainiert wurde nur ein linearer Head von 768 auf vier Klassen.

| Merkmal | Tatsächliche Einstellung |
| --- | --- |
| Modell | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Aufgabe | globale Vier-Klassen-Bildklassifikation |
| Backbone | eingefroren |
| Head | linear, 768 auf 4 |
| Eingabegröße | 224 x 224 Pixel |
| Batchgröße | 16 |
| Epochen | 50 |
| Checkpoint-Kriterium | Validation Macro-F1 |
| bester Checkpoint | Epoche 26 |
| Optimierer | AdamW |
| Lernrate | 0,001 |
| Weight Decay | 0,0001 |
| AMP | aktiviert |
| Seed | 42 |
| Config | `configs/experiments/dinov3_linear_head.yaml` |
| Skript | `scripts/train_dinov3_head.py` |

Der Lauf verwendete kein Early Stopping. Lokale Artefakte liegen unter
`outputs/global_classification/dinov3_linear_head_bmw25_seed42/`, darunter
Run-Metadaten, Trainingslog, `checkpoints/best_head.pt`,
`checkpoints/last_head.pt`, Validierungsprädiktionen und Metriken.

### DeiT-Tiny from scratch

`deit_tiny_patch16_224` wurde über `timm` mit `pretrained=false` erzeugt. Das
Modell dient als ViT-Kontrollarchitektur ohne externe Vortrainingsinformation.

| Merkmal | Tatsächliche Einstellung |
| --- | --- |
| Modell | `deit_tiny_patch16_224` |
| Quelle | `timm` |
| Aufgabe | globale Vier-Klassen-Bildklassifikation |
| Vortraining | keines |
| Eingabegröße | 224 x 224 Pixel |
| Batchgröße | 16 |
| maximales Epochenbudget | 150 |
| protokollierte Epochen | 92 |
| bester Checkpoint | Epoche 68 nach Validation Macro-F1 |
| Optimierer | AdamW |
| Lernrate | 0,0005 |
| Weight Decay | 0,05 |
| AMP | aktiviert |
| Seed | 42 |
| Config | `configs/experiments/deit_tiny_scratch.yaml` |
| Skript | `scripts/train_deit_tiny.py` |

Nach dem besten Checkpoint wurden 24 Epochen ohne weitere Verbesserung
protokolliert. Ein vollständig dokumentierter regulärer
Early-Stopping-Abschluss liegt nicht vor. Die vorhandenen Logs enden nach
Epoche 92; der gespeicherte beste Checkpoint wurde anschließend ausschließlich
auf Validation ausgewertet. Lokale Artefakte liegen unter
`outputs/global_classification/deit_tiny_scratch_bmw25_seed42/`.

### DINOv3 mit partiellem Fine-Tuning

Die Optimierungsvariante trainierte den linearen Head, die letzten zwei
Transformer-Blöcke und die finale Norm. Patch Embedding und frühere Blöcke
blieben eingefroren.

| Merkmal | Tatsächliche Einstellung |
| --- | --- |
| Modell | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Aufgabe | globale Vier-Klassen-Bildklassifikation |
| trainierbare Backbone-Module | `model.layer.10`, `model.layer.11`, finale Norm |
| Head | linear, 768 auf 4 |
| Gesamtparameter einschließlich Head | 85.663.492 |
| trainierbare Parameter einschließlich Head | 14.181.892 |
| Eingabegröße | 224 x 224 Pixel |
| Batchgröße | 8 |
| maximales Epochenbudget | 30 |
| tatsächlich gelaufene Epochen | 14 |
| Early Stopping | Patience 10, ausgelöst |
| bester Checkpoint | Epoche 4 nach Validation Macro-F1 |
| Optimierer | AdamW |
| Backbone-Lernrate | 0,00001 |
| Head-Lernrate | 0,0005 |
| Weight Decay | 0,05 |
| AMP | aktiviert |
| Seed | 42 |
| Config | `configs/experiments/dinov3_partial_finetune_last2.yaml` |
| Skript | `scripts/train_dinov3_partial_finetune.py` |

Der Lauf liegt unter
`outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/`.
Er enthält lokale Metadaten, Logs, den besten und letzten Modellcheckpoint,
Validierungsprädiktionen, Metriken und eine Confusion Matrix.

## DINOv3-Preprocessing

Globale DINOv3-Bilder werden mit derselben deterministischen Vorbereitung wie
im Training verarbeitet:

```text
EXIF-Transpose
→ RGB
→ seitenverhältnistreues BICUBIC-Resize
→ schwarzes Padding auf 224 × 224
→ DINOv3-Processor
```

Diese Reihenfolge gilt auch für die technisch korrigierte finale
DINOv3-Auswertung. Augmentierungen werden nur im Training eingesetzt.

## Lokale rechteckbasierte Experimente

### Direkte DINOv3-Regionenauswertung

Die direkte Regionenauswertung verwendet den besten globalen
DINOv3-Partial-Fine-Tuning-Checkpoint ausschließlich für Inferenz. Aus den
manuellen CVAT-Bounding-Boxes werden 224 x 224 Pixel große Eingaben erzeugt.
Auf Validation wurden fünf Crop-Varianten verglichen: quadratisches Padding
mit Kontextmargen 0,0, 0,1, 0,25 und 0,5 sowie direktes Stretch-Resize ohne
Kontextmarge. Es fand kein Training statt.

| Merkmal | Einstellung |
| --- | --- |
| Config | `configs/experiments/dinov3_region_eval.yaml` |
| Skript | `scripts/evaluate_dinov3_regions.py` |
| Checkpoint | bester globaler DINOv3-Partial-Fine-Tuning-Checkpoint |
| Eingabegröße | 224 x 224 Pixel |
| Auswahlmetrik | Validation Macro-F1, Tie-Breaker Balanced Accuracy |
| lokale Artefakte | Predictions, Metriken, Ablationstabellen und Visualisierungen unter `outputs/region_analysis/` |

### DINOv3-Region-Head mit vier Klassen

Für die Hauptvariante blieb der DINOv3-Backbone aus dem Partial-Fine-Tuning-
Checkpoint eingefroren. Nur ein kleiner MLP-Head wurde auf den CVAT-Regionen
trainiert.

| Merkmal | Tatsächliche Einstellung |
| --- | --- |
| Aufgabe | lokale Vier-Klassen-Klassifikation auf Bounding-Box-Crops |
| Daten | 147 Train- und 35 Validierungsregionen |
| Backbone | eingefroren |
| Head | Linear 768 auf 128, ReLU, Dropout 0,2, Linear 128 auf 4 |
| trainierbare Parameter | 98.948 |
| Crop-Modus | `stretch_resize`, Kontextmarge 0,0 |
| Eingabegröße | 224 x 224 Pixel |
| Batchgröße | 16 |
| maximales Epochenbudget | 100 |
| tatsächlich gelaufene Epochen | 25 |
| Early Stopping | Patience 15, ausgelöst |
| bester Checkpoint | Epoche 10 nach Validation Macro-F1 |
| Optimierer | AdamW |
| Lernrate | 0,001 |
| Weight Decay | 0,01 |
| Klassengewichte | aktiviert |
| Seed | 42 |
| Config | `configs/experiments/dinov3_region_head.yaml` |
| Skript | `scripts/train_dinov3_region_head.py` |

Die Trainingsaugmentation bestand aus leichten Helligkeits- und
Kontraständerungen sowie horizontalen und vertikalen Spiegelungen. Validation
wurde deterministisch verarbeitet. Lokale Artefakte liegen unter
`outputs/region_analysis/dinov3_region_head_bmw25_seed42/`.

### DINOv3-Region-Head mit fünf Klassen

Das Zusatzexperiment ergänzt `Nicht_bewertbar` als fünfte Klasse. Backbone,
Head-Aufbau und Optimierung entsprechen der Vier-Klassen-Variante; die letzte
lineare Schicht besitzt fünf Ausgänge.

| Merkmal | Tatsächliche Einstellung |
| --- | --- |
| Aufgabe | lokale Fünf-Klassen-Klassifikation auf Bounding-Box-Crops |
| Daten | 181 Train- und 42 Validierungsregionen |
| Backbone | eingefroren |
| Head | Linear 768 auf 128, ReLU, Dropout 0,2, Linear 128 auf 5 |
| trainierbare Parameter | 99.077 |
| Crop-Modus | `stretch_resize`, Kontextmarge 0,0 |
| Eingabegröße | 224 x 224 Pixel |
| Batchgröße | 16 |
| maximales Epochenbudget | 100 |
| tatsächlich gelaufene Epochen | 17 |
| Early Stopping | Patience 15, ausgelöst |
| bester Checkpoint | Epoche 2 nach Validation Macro-F1 |
| Optimierer | AdamW |
| Lernrate | 0,001 |
| Weight Decay | 0,01 |
| Klassengewichte | aktiviert |
| Seed | 42 |
| Config | `configs/experiments/dinov3_region_head_5class.yaml` |
| Skript | `scripts/train_dinov3_region_head.py` |

Die lokalen Ausgaben liegen unter
`outputs/region_analysis/dinov3_region_head_5class_bmw25_seed42/`. Beide
Region-Heads speichern Run-Metadaten, Trainingslogs, den besten und letzten
Checkpoint, Validierungsprädiktionen, Metriken, Confusion Matrices und
optionale Visualisierungen. Diese Dateien sind nicht versioniert.

## Verwandte Dokumente

Die methodische Einordnung steht in der [Methodik](methodology.md); die
zugehörigen Ergebnisse und Auswahlentscheidungen in der
[Validierungsauswertung](validation_results.md). Ausführbare Befehle und
lokale Voraussetzungen sind in der
[Reproduzierbarkeitsanleitung](reproducibility.md) zusammengefasst.
Lizenzangaben bleiben in der [Lizenzdokumentation](model_licenses.md)
kanonisch erhalten.
