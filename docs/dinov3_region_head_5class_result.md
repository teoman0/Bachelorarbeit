# DINOv3-Region-Head-5-Klassen-Ergebnis

Stand: 2026-07-20

## Ziel

Das Zusatzexperiment prueft, ob die Sonderklasse `Nicht_bewertbar` sinnvoll in
die lokale Regionenauswertung integriert werden kann. Die Analyse bleibt eine
regionenbasierte Klassifikation auf CVAT-Bounding-Box-Regionen und ist keine
echte semantische Segmentierung.

## Experiment

| Merkmal | Wert |
| --- | --- |
| Name | `dinov3_region_head_5class_bmw25_seed42` |
| Grundlage | DINOv3 Partial Fine-Tuning |
| Aufgabe | lokale 5-Klassen-Klassifikation auf CVAT-Bounding-Box-Regionen |
| Backbone | frozen |
| Trainierbar | nur 5-Klassen-Region-Head |
| Head-Parameter | 99,077 |
| Crop-Modus | `stretch_resize` |
| Testset | nicht verwendet |

Klassen:

- Erste Bearbeitungsstufe Viertel
- Zweite Bearbeitungsstufe Viertel
- Fraeszustand Viertel
- Finaler Zustand Viertel
- `Nicht_bewertbar`

## Daten

| Datenbereich | Anzahl |
| --- | ---: |
| Train-Regionen | 181 |
| Val-Regionen | 42 |
| `Nicht_bewertbar` | als fuenfte Klasse enthalten |
| Testregionen | ausgeschlossen |

Es wurden keine neuen Splits erzeugt. Das bestehende `region_annotations.csv`
blieb verbindlich; Train-Regionen wurden nur zum Training und Val-Regionen zur
Checkpoint-Auswahl, Bewertung und Visualisierung genutzt.

## Training

| Parameter | Wert |
| --- | --- |
| Laufzeit | ca. 4 min 27 s |
| Epochen gelaufen | 17 |
| Early Stopping | ja |
| Bester Epoch nach Validation Macro-F1 | 2 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Batch Size | 16 |
| Patience | 15 |
| Seed | 42 |

## 5-Klassen-Val-Metriken

| Metrik | Wert |
| --- | ---: |
| Accuracy | 0.7142857143 |
| Balanced Accuracy | 0.7517732268 |
| Macro-F1 | 0.7160455487 |

Diese Werte sind Validierungsergebnisse. Das Testset blieb fuer die finale
Bewertung reserviert.

## Klassenweise Metriken

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe Viertel | 0.9000 | 0.6923 | 0.7826 |
| Zweite Bearbeitungsstufe Viertel | 0.6154 | 0.7273 | 0.6667 |
| Fraeszustand Viertel | 0.7143 | 0.6250 | 0.6667 |
| Finaler Zustand Viertel | 0.6000 | 1.0000 | 0.7500 |
| `Nicht_bewertbar` | 0.7143 | 0.7143 | 0.7143 |

## Confusion Matrix

Zeilen/Spalten: Erste, Zweite, Fraeszustand, Final, Nicht_bewertbar

```text
Erste Bearbeitungsstufe       9  3  0  0  1
Zweite Bearbeitungsstufe      1  8  0  2  0
Fraeszustand                  0  2  5  0  1
Finaler Zustand               0  0  0  3  0
Nicht_bewertbar               0  0  2  0  5
```

## Nicht Bewertbar

- 5 von 7 Val-Regionen korrekt erkannt
- Precision: 0.7143
- Recall: 0.7143
- F1: 0.7143

## Vergleich Zum 4-Klassen-Region-Head

| Metrik | 4-Klassen-Head | 5-Klassen-Head |
| --- | ---: | ---: |
| Accuracy | 0.7714 | 0.7143 |
| Balanced Accuracy | 0.8116 | 0.7518 |
| Macro-F1 | 0.7768 | 0.7160 |

## Interpretation

Der 5-Klassen-Head ist schwaecher als der 4-Klassen-Head. Die Sonderklasse
`Nicht_bewertbar` kann grundsaetzlich erkannt werden, wurde im Val-Split aber
nur durch 7 Regionen repraesentiert.

Die zusaetzliche Klasse erhoeht die Schwierigkeit der lokalen Klassifikation
und reduziert die Gesamtleistung. Fuer die Hauptauswertung bleibt daher der
4-Klassen-Region-Head der staerkere lokale Schleifgradklassifikator.

Der 5-Klassen-Head wird als ergaenzendes praxisnahes Zusatzexperiment
dokumentiert, weil eine Ausschlussklasse fuer nicht eindeutig bewertbare
Bereiche in industriellen Anwendungen sinnvoll sein kann. Aufgrund der kleinen
Datenbasis und des sehr fruehen besten Epochs 2 besteht ein erhoehtes
Overfitting-Risiko.

## Lokale Artefakte

Die lokalen Artefakte bleiben ignored und duerfen nicht committed werden:

```text
outputs/region_analysis/dinov3_region_head_5class_bmw25_seed42/
```
