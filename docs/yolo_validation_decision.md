# YOLO-Validierungsentscheidung

Stand: 2026-07-16

## Ziel

Diese Notiz dokumentiert die Validierungsentscheidung fuer die
YOLOv11-cls-Baseline im globalen Bildklassifikationsvergleich. Die
Entscheidung betrifft nur die Modellauswahl anhand des Validierungssplits.
Sie ist keine finale Testbewertung.

## Gepruefte Varianten

Beide Varianten wurden mit derselben lokalen YOLO-Dataset-Struktur aus dem
versionierten Split-Manifest trainiert und anschliessend auf dem
Validierungssplit ausgewertet:

- `yolo11n-cls`, `imgsz=320`, `batch=16`, `seed=42`;
- `yolo11s-cls`, `imgsz=320`, `batch=16`, `seed=42`.

Verwendet wurden ausschliesslich Train und Validation. Das Testset blieb
unberuehrt und wird erst fuer die spaetere finale Evaluation genutzt.

## Vergleich

| Variante | Bester Epoch laut Ultralytics | Accuracy (Val) | Balanced Accuracy (Val) | Macro-F1 (Val) |
| --- | ---: | ---: | ---: | ---: |
| `yolo11n-cls` | 42 | 0.9580318379 | 0.9477213967 | 0.9534721203 |
| `yolo11s-cls` | 30 | 0.9450072359 | 0.9395723401 | 0.9437327438 |

## Entscheidung

`yolo11n-cls` bleibt die YOLO-Baseline fuer die spaetere finale
Testbewertung. Die kleinere Variante erzielt auf dem Validierungssplit die
besseren Werte fuer Accuracy, Balanced Accuracy und Macro-F1.

`yolo11s-cls` wird in dieser Konfiguration nicht weiterverfolgt, weil das
groessere Modell auf dem Validierungssplit keinen Mehrwert zeigt und in allen
drei dokumentierten Auswahlmetriken schlechter abschneidet.

## Testset-Hinweis

Die hier dokumentierten Zahlen sind Validierungsergebnisse. Das Testset wurde
nicht fuer Training, Early Stopping, Checkpoint-Auswahl, Hyperparameter- oder
Modellentscheidung genutzt. Die finale Testbewertung erfolgt erst spaeter mit
dem ausgewaehlten Modell.
