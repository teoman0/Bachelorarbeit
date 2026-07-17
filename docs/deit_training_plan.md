# DeiT-Tiny-Trainingsplan

Stand: 2026-07-17

## Ziel

Dieser Plan beschreibt den vorgesehenen DeiT-Tiny-from-scratch-Workflow fuer
die globale Bildklassifikation der vier Schleifgradklassen. Er dokumentiert
die geplante Kontrollarchitektur, aber keine finalen Ergebnisse.

## Modellwahl

Verwendet wird `deit_tiny_patch16_224` aus `timm`. DeiT-Tiny ist klein genug
fuer die lokale GPU und passt zum gemeinsamen ViT-Eingabeformat `224 x 224`.
Es dient als kompakte ViT-Architekturkontrolle gegenueber den vortrainierten
oder praxisnahen Baselines.

## Warum pretrained=false

Der Lauf verwendet `pretrained=false`, damit keine externe
Vortrainingsinformation in diese Kontrollvariante einfliesst. Dadurch ist der
Ansatz nicht direkt mit DINOv3 gleichzusetzen: DINOv3 bewertet die Qualitaet
vortrainierter Repraesentationen, waehrend DeiT-Tiny from scratch prueft, wie
weit eine kleine ViT-Architektur mit dem vorhandenen Datensatz allein kommt.

## Erwartete Schwierigkeit

Ein Transformer-Modell ohne Vortraining kann bei einem begrenzten Datensatz
schwerer stabil zu trainieren sein. Der Lauf ist deshalb methodisch eher als
from-scratch-Untergrenze bzw. Architekturkontrolle zu verstehen, nicht als
direkter Ersatz fuer DINOv3.

## Artefakte und Split-Regeln

Trainiert und validiert wird ausschliesslich mit dem bestehenden gruppierten
Split-Manifest. Das Testset bleibt bis zur finalen Evaluation unberuehrt.
Lokale Outputs, Predictions, Metriken, Gewichte und Checkpoints bleiben unter
ignorierten Pfaden wie `outputs/`, `runs/`, `weights/` oder `checkpoints/`.

## Offene Punkte vor echtem Training

- `timm` in der lokalen Python-Umgebung pruefen.
- `--dry-run`, `--check-model` und `--smoke-test` erfolgreich ausfuehren.
- Bei VRAM-Problemen Batch Size `8` als Fallback verwenden.
- Erst nach erfolgreichem Techniktest den echten from-scratch-Trainingslauf
  starten.
- Finale Ergebnisse spaeter separat dokumentieren.
