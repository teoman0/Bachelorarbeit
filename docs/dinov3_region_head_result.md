# DINOv3-Region-Head-Ergebnis

Stand: 2026-07-20

## Ziel

Der Lauf prueft eine regionenbasierte lokale Klassifikation auf manuell
annotierten CVAT-Bounding-Box-Regionen. Die Analyse ist keine echte
semantische Segmentierung, sondern eine rechteckbasierte Auswertung lokaler
Bildbereiche.

## Experiment

| Merkmal | Wert |
| --- | --- |
| Name | `dinov3_region_head_bmw25_seed42` |
| Grundlage | DINOv3 Partial Fine-Tuning |
| Aufgabe | lokale 4-Klassen-Klassifikation auf CVAT-Bounding-Box-Regionen |
| Backbone | frozen |
| Trainierbar | lokaler Region-Head |
| Feature-Dimension | 768 |
| Trainierbare Head-Parameter | 98,948 |
| Crop-Modus | `stretch_resize` |
| Context Margin | `0.0` |
| Testset | nicht verwendet |
| Sonderklasse | `Nicht_bewertbar` aus Training und 4-Klassen-Metrik ausgeschlossen |

## Daten

| Datenbereich | Anzahl |
| --- | ---: |
| Train-Regionen 4-Klassen | 147 |
| Val-Regionen 4-Klassen | 35 |
| `Nicht_bewertbar` | ausgeschlossen |
| Testregionen | ausgeschlossen |

Es wurden keine neuen Splits erzeugt. Grundlage war die vorhandene
`region_annotations.csv`, die aus dem bestehenden grouped split manifest
abgeleitet wurde.

## Training

| Parameter | Wert |
| --- | --- |
| Laufzeit | ca. 5 min 26 s |
| Epochen gelaufen | 25 |
| Early Stopping | ja |
| Bester Epoch nach Validation Macro-F1 | 10 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Batch Size | 16 |
| Patience | 15 |
| Seed | 42 |

## Beste Val-Metriken

| Metrik | Wert |
| --- | ---: |
| Accuracy | 0.7714285714 |
| Balanced Accuracy | 0.8116258741 |
| Macro-F1 | 0.7767857143 |

Diese Werte sind Validierungsergebnisse. Das Testset blieb fuer die finale
Bewertung reserviert.

## Klassenweise Metriken

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe Viertel | 0.9091 | 0.7692 | 0.8333 |
| Finaler Zustand Viertel | 0.6000 | 1.0000 | 0.7500 |
| Fraeszustand Viertel | 1.0000 | 0.7500 | 0.8571 |
| Zweite Bearbeitungsstufe Viertel | 0.6154 | 0.7273 | 0.6667 |

## Confusion Matrix

Zeilen = True Label, Spalten = Prediction: Erste, Final, Fraeszustand, Zweite

```text
Erste Bearbeitungsstufe     10  0  0  3
Finaler Zustand              0  3  0  0
Fraeszustand                  0  0  6  2
Zweite Bearbeitungsstufe      1  2  0  8
```

## Haeufigste Verwechslungen

- Erste Bearbeitungsstufe -> Zweite Bearbeitungsstufe: 3
- Fraeszustand -> Zweite Bearbeitungsstufe: 2
- Zweite Bearbeitungsstufe -> Finaler Zustand: 2
- Zweite Bearbeitungsstufe -> Erste Bearbeitungsstufe: 1

## Vergleich Zur Direkten Region-Inferenz

| Metrik | Direkte Region-Inferenz | Region-Head |
| --- | ---: | ---: |
| Accuracy | 0.6857 | 0.7714 |
| Balanced Accuracy | 0.7489 | 0.8116 |
| Macro-F1 | 0.7233 | 0.7768 |

## Interpretation

Der lokale Region-Head verbessert die Auswertung gegenueber der direkten
Anwendung des globalen DINOv3-Modells auf Region-Crops. Das zeigt, dass eine
gezielte Anpassung an die manuell annotierten lokalen Regionen sinnvoll ist.

Die lokale Leistung bleibt unter der globalen Bildklassifikation. Plausible
Gruende sind die kleine Anzahl annotierter Regionen, der Verlust des
Bildkontexts und die rechteckbasierten statt pixelgenauen Annotationen.

Der Verlauf zeigt ein Overfitting-Risiko: Der Train-Loss faellt stark, waehrend
der beste Val-Wert bereits in Epoche 10 erreicht wird. Die Ergebnisse sind
daher als ergaenzende lokale Plausibilisierung zu verstehen, nicht als
vollwertige Segmentierungsbewertung.

Fuer eine echte semantische Segmentierung waeren pixelgenaue Masken oder
Polygone sowie ein groesserer lokal annotierter Datensatz erforderlich.

## Visualisierungen

| Artefakt | Anzahl |
| --- | ---: |
| Ground Truth | 12 Bilder |
| Predictions | 12 Bilder |
| Comparison | 12 Bilder |
| Crops | 35 Bilder |

Index:

```text
outputs/region_analysis/dinov3_region_head_bmw25_seed42/visualizations/region_visualization_index.csv
```

## Lokale Artefakte

Die folgenden Dateien wurden lokal erzeugt und duerfen nicht committed werden:

- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/run_metadata.json`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/training_log.csv`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/training_metrics.csv`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/checkpoints/best_model.pt`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/checkpoints/last_model.pt`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/predictions_regions_val.csv`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/val_region_metrics.json`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/val_region_metrics.csv`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/confusion_matrix_val_regions.csv`
- `outputs/region_analysis/dinov3_region_head_bmw25_seed42/visualizations/`
