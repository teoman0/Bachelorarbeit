# DINOv3-Validierungsergebnis

Stand: 2026-07-16

## Ziel

Diese Datei dokumentiert den ersten ernsthaften DINOv3-Validierungslauf fuer
die globale Bildklassifikation der vier Schleifgradklassen. Die Zahlen sind
Validierungsergebnisse und keine finalen Testwerte.

## Konfiguration

| Parameter | Wert |
| --- | --- |
| Modell | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Backbone | frozen |
| trainierbare Backbone-Parameter | 0 |
| Head | linear, `768 -> 4` |
| Eingabegroesse | `224 x 224` |
| Batch Size | 16 |
| Epochen | 50 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Seed | 42 |
| Device | `cuda:0`, NVIDIA GeForce RTX 4060 Ti |
| Bester Epoch | 26 nach Validation Macro-F1 |

Der DINOv3-Backbone blieb eingefroren. Trainiert wurde nur der lineare
Klassifikationskopf. Verwendet wurden ausschliesslich Train und Validation
aus dem bestehenden gruppierten Split-Manifest:

- Train: 3225 Bilder
- Validation: 691 Bilder

Das Testset blieb unberuehrt und wird erst fuer die finale Evaluation genutzt.

## Gesamtmetriken

| Modell | Accuracy (Val) | Balanced Accuracy (Val) | Macro-F1 (Val) |
| --- | ---: | ---: | ---: |
| DINOv3 ViT-B/16 + linear head | 0.9551374819 | 0.9517583436 | 0.9555180393 |

## Klassenweise Metriken

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungssufe Viertel | 0.977011 | 0.923913 | 0.949721 |
| Finaler Zustand Viertel | 0.937500 | 0.963303 | 0.950226 |
| Fräszustand Viertel | 0.994444 | 0.994444 | 0.994444 |
| Zweite Bearbeitungsstufe Viertel | 0.930000 | 0.925373 | 0.927681 |

## Confusion Matrix

Zeilen sind wahre Klassen, Spalten sind vorhergesagte Klassen.

| True \ Pred | Erste Bearbeitungssufe | Finaler Zustand | Fräszustand | Zweite Bearbeitungsstufe |
| --- | ---: | ---: | ---: | ---: |
| Erste Bearbeitungssufe | 85 | 1 | 0 | 6 |
| Finaler Zustand | 0 | 210 | 0 | 8 |
| Fräszustand | 1 | 0 | 179 | 0 |
| Zweite Bearbeitungsstufe | 1 | 13 | 1 | 186 |

## Vergleich zu YOLOv11n-cls

| Modell | Accuracy (Val) | Balanced Accuracy (Val) | Macro-F1 (Val) |
| --- | ---: | ---: | ---: |
| YOLOv11n-cls | 0.9580318379 | 0.9477213967 | 0.9534721203 |
| DINOv3 ViT-B/16 + linear head | 0.9551374819 | 0.9517583436 | 0.9555180393 |

YOLOv11n-cls erreicht auf dem Validierungssplit eine minimal hoehere
Accuracy. DINOv3 erreicht dagegen eine leicht hoehere Balanced Accuracy und
einen leicht hoeheren Macro-F1. Besonders bei der kleinsten Klasse
`Erste Bearbeitungssufe Viertel` verbessert DINOv3 Recall und F1 gegenueber
der YOLO-Baseline.

## Entscheidung

DINOv3 ViT-B/16 mit eingefrorenem Backbone und linearem Head bleibt der
primaere DINOv3-Kandidat fuer die spaetere finale Evaluation. Die finale
Testbewertung erfolgt separat und wird erst nach Abschluss der
Validierungsentscheidungen durchgefuehrt.
