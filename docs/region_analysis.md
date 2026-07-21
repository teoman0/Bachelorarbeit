# Rechteckbasierte Regionenanalyse

## Zweck und Geltungsbereich

Dieses Dokument ist die kanonische Beschreibung der lokalen
rechteckbasierten Regionenanalyse, ihrer CVAT-Datengrundlage, Crop-Strategien,
Region-Heads und Visualisierungen. Die Aufgabe ist eine lokale Klassifikation
und keine semantische Segmentierung.

## Ziel und Abgrenzung

Die lokale Analyse untersucht, ob global gelernte DINOv3-Repräsentationen
auch in manuell markierten Teilbereichen metallischer Oberflächen plausible
Schleifgradvorhersagen ermöglichen. Grundlage sind rechteckige
CVAT-Annotationen. Die Auswertung ist damit eine Klassifikation von
Bounding-Box-Crops und keine semantische Segmentierung. Es liegen weder
Polygone noch pixelgenaue Masken vor; Segmentierungsmetriken wie IoU oder
Pixel-Accuracy sind daher nicht anwendbar.

## Annotationen

Der lokale CVAT-Export liegt im JSON-Format vor und wird zusammen mit
Frame-Metadaten, Labeldefinitionen und einem lokalen Bildmanifest eingelesen.
Er umfasst 88 annotierte Bilder und 302 rechteckige Regionen.

| CVAT-Label | Wissenschaftliche Bezeichnung | Regionen |
| --- | --- | ---: |
| `Stufe_1` | Erste Bearbeitungsstufe | 72 |
| `Stufe_2` | Zweite Bearbeitungsstufe | 80 |
| `Fraeszustand` | Fräszustand | 47 |
| `Final` | Finaler Zustand | 44 |
| `Nicht_bewertbar` | Nicht_bewertbar | 59 |
| **Gesamt** |  | **302** |

`Nicht_bewertbar` ist eine lokale Sonderklasse und gehört nicht zu den vier
globalen Schleifgradklassen. In der Vier-Klassen-Auswertung wird sie
ausgeschlossen; das separate Fünf-Klassen-Experiment nimmt sie als reguläre
Zielklasse auf.

## Zuordnung zum gruppierten Split

`scripts/prepare_cvat_region_annotations.py` entfernt den Präfix der manuellen
Annotation und leitet aus dem ursprünglichen Bildnamen dieselbe Gruppen-ID wie
im globalen Split ab. Dadurch übernehmen Regionen den bereits festgelegten
Split ihres Ursprungsbildes. Es werden keine neuen Splits erzeugt.

| Zuordnung | Annotierte Bilder | Regionen |
| --- | ---: | ---: |
| Training | 59 | 181 |
| Validation | 12 | 42 |
| Test | 13 | 58 |
| nicht zuordenbar | 4 | 21 |
| **Gesamt** | **88** | **302** |

| Zuordnung | Erste | Zweite | Fräszustand | Final | Nicht_bewertbar | Gesamt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 41 | 51 | 32 | 23 | 34 | 181 |
| Validation | 13 | 11 | 8 | 3 | 7 | 42 |
| Test | 8 | 16 | 7 | 17 | 10 | 58 |
| nicht zuordenbar | 10 | 2 | 0 | 1 | 8 | 21 |

Nicht zuordenbare Regionen werden markiert und von Training und Bewertung
ausgeschlossen. Eine Bounding Box musste an die Bildgrenzen geclippt werden;
fehlende Quelldateien wurden nicht festgestellt. Die Testregionen waren
während der Entwicklung inventarisiert, aber von Training, Crop-Auswahl,
Checkpoint-Auswahl und Validierungsmetriken ausgeschlossen. Ihre finale
Auswertung ist separat in `docs/final_test_evaluation_result.md` dokumentiert.

## Regionentabelle

Die lokal erzeugte Regionentabelle enthält für jede Box unter anderem eine
Region-ID, Quellbild und Originalbildname, Gruppen-ID, Split, ursprüngliches
und gemapptes Label, Bildgröße, geclippte Boxkoordinaten, Boxfläche,
Flächenanteil, Matching-Status und Ausschlussgrund. Absolute lokale Pfade
werden nicht gespeichert.

Für die Entwicklungsphase standen nach Ausschluss der Test- und nicht
zuordenbaren Regionen 181 Trainings- und 42 Validierungsregionen zur
Verfügung, insgesamt also 223 Regionen. Ohne `Nicht_bewertbar` verblieben
147 Trainings- und 35
Validierungsregionen für die Vier-Klassen-Aufgabe. Regionentabellen und daraus
abgeleitete Crops bleiben lokale, ignorierte Artefakte.

## Direkte DINOv3-Inferenz

Zunächst wurde der beste globale DINOv3-Partial-Fine-Tuning-Checkpoint direkt
auf den rechteckigen Crops angewendet. Das Modell wurde dabei nicht weiter
trainiert. Die Validation-Ablation verglich folgende Eingabestrategien:

- quadratisches schwarzes Padding mit Kontextmargen 0,0, 0,1, 0,25 und 0,5;
- direktes Skalieren der Bounding Box auf 224 x 224 Pixel ohne Kontextmarge.

| Crop-Strategie | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| `pad_square`, Kontext 0,00 | 0,6571 | 0,7107 | 0,6506 |
| `pad_square`, Kontext 0,10 | 0,6571 | 0,7107 | 0,6742 |
| `pad_square`, Kontext 0,25 | 0,6286 | 0,7015 | 0,6622 |
| `pad_square`, Kontext 0,50 | 0,6571 | 0,6882 | 0,6474 |
| `stretch_resize`, Kontext 0,00 | **0,6857** | **0,7489** | **0,7233** |

Das direkte Skalieren ohne Kontextmarge erreichte den höchsten Validation
Macro-F1 und wurde als Eingabe für die Region-Heads verwendet. Die schlechtere
Leistung gegenüber der globalen Bildklassifikation zeigt, dass ein auf
Gesamtbildern trainierter Klassifikator nicht ohne Weiteres auf kleine lokale
Ausschnitte übertragbar ist.

Confusion Matrix der besten direkten Variante, Zeilen = Referenz und Spalten =
Vorhersage. Reihenfolge: Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe,
Fräszustand, Finaler Zustand.

```text
[6,  7, 0, 0]
[0, 10, 0, 1]
[1,  2, 5, 0]
[0,  0, 0, 3]
```

## Region-Heads

Für beide Region-Heads wird der DINOv3-Backbone aus dem globalen
Partial-Fine-Tuning-Checkpoint geladen und vollständig eingefroren. Trainiert
wird ausschließlich ein MLP:

```text
Linear 768 -> 128
ReLU
Dropout 0,2
Linear 128 -> 4 beziehungsweise 5
```

Im tatsächlich ausgeführten Training waren leichte Helligkeits- und
Kontraständerungen mit Stärke 0,08 sowie horizontale und vertikale
Spiegelungen mit Wahrscheinlichkeit 0,5 aktiviert. Die Spiegelungen setzen
voraus, dass die Schleifgradklasse gegenüber der Bildorientierung invariant
bleibt. Da Schleifspuren richtungsabhängig erscheinen können, ist diese
Annahme eine methodische Einschränkung und bei der Interpretation zu
berücksichtigen.

### Vier-Klassen-Hauptvariante

Der Vier-Klassen-Head besitzt 98.948 trainierbare Parameter. Er erreichte auf
35 Validierungsregionen eine Accuracy von 0,7714, eine Balanced Accuracy von
0,8116 und einen Macro-F1 von 0,7768. Damit übertraf er die direkte
Regioneninferenz und bildet die lokale Hauptvariante.

Confusion Matrix, Zeilen = Referenz und Spalten = Vorhersage. Reihenfolge:
Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe, Fräszustand, Finaler
Zustand.

```text
[10, 3, 0, 0]
[ 1, 8, 0, 2]
[ 0, 2, 6, 0]
[ 0, 0, 0, 3]
```

### Fünf-Klassen-Zusatzexperiment

Der Fünf-Klassen-Head besitzt 99.077 trainierbare Parameter und schließt
`Nicht_bewertbar` als fünfte Klasse ein. Auf 42 Validierungsregionen erreichte
er eine Accuracy von 0,7143, eine Balanced Accuracy von 0,7518 und einen
Macro-F1 von 0,7160. Fünf der sieben `Nicht_bewertbar`-Regionen wurden korrekt
erkannt. Das Zusatzexperiment zeigt die grundsätzliche Erkennbarkeit der
Sonderklasse, ist wegen ihrer geringen Fallzahl jedoch nur eingeschränkt
belastbar.

Confusion Matrix, Zeilen = Referenz und Spalten = Vorhersage. Reihenfolge:
Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe, Fräszustand, Finaler
Zustand, Nicht_bewertbar.

```text
[9, 3, 0, 0, 1]
[1, 8, 0, 2, 0]
[0, 2, 5, 0, 1]
[0, 0, 0, 3, 0]
[0, 0, 2, 0, 5]
```

## Visualisierungen

Die lokalen Visualisierungen bestehen aus vier Artefakttypen:

- `ground_truth/`: Originalbilder mit blauen Boxen; `GT:` bezeichnet das
  Referenzlabel;
- `predictions/`: Originalbilder mit dem Präfix `P:` für die vorhergesagte
  Klasse; die Zahl in Klammern ist die maximale Softmax-Wahrscheinlichkeit;
- `comparison/`: Ground Truth links und Prediction rechts;
- `crops/`: einzelne Eingabecrops mit Label, Vorhersage, Confidence und
  Korrektheitsstatus im Dateinamen.

In Prediction-Overlays kennzeichnet Grün eine korrekte und Rot eine falsche
Vorhersage der bewerteten Schleifgradklassen. `Nicht_bewertbar` wird neutral
grau dargestellt. Der Ground-Truth-Overlay verwendet unabhängig vom Ergebnis
Blau. Die Datei `region_visualization_index.csv` verbindet Bild-ID, Region-ID,
wahres Label, vorhergesagtes Label, Confidence, Korrektheitsstatus sowie die
zugehörigen Vergleichs- und Crop-Dateien.

## Grenzen

Die Aussagekraft der lokalen Experimente wird vor allem durch die kleine Zahl
annotierter Bilder und Regionen begrenzt. Mehrere Regionen stammen aus
demselben Ursprungsbild und sind deshalb nicht als unabhängige Bildstichprobe
zu interpretieren. Rechtecke enthalten je nach Annotation zusätzlichen
Kontext oder Randbereiche und bilden genaue Materialgrenzen nicht ab. Beim
Cropping geht zugleich globaler Bildkontext verloren. Die Ergebnisse sind
daher eine ergänzende lokale Plausibilisierung der Schleifgradklassifikation,
nicht der Nachweis einer pixelgenauen Oberflächenlokalisierung.

## Verwandte Dokumente

Methodische Bewertungsregeln stehen in der [Methodik](methodology.md). Die
vollständige Validierungsentscheidung dokumentiert die
[Validierungsauswertung](validation_results.md), die technische Ausführung die
[Reproduzierbarkeitsanleitung](reproducibility.md) und die finale Bewertung
die [finale Testauswertung](final_test_evaluation_result.md).
