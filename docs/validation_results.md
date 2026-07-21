# Validierungsergebnisse

## Zweck und Geltungsbereich

Dieses Dokument ist die kanonische Quelle für Validierungsergebnisse,
Checkpoint-Auswahl und Modellentscheidungen. Alle Zahlen stammen aus
Validation und sind strikt von den finalen Testwerten in
`docs/final_test_evaluation_result.md` getrennt.

## Einordnung

Dieses Dokument enthält ausschließlich Ergebnisse des Validierungssplits.
Sie wurden während der Modellentwicklung für Checkpoint-Auswahl,
Variantenvergleich und die Festlegung der abschließend bewerteten Modelle
verwendet. Finale Testwerte sind hiervon getrennt und werden nur in
`docs/final_test_evaluation_result.md` berichtet.

Alle globalen Modelle verwenden denselben gruppierten Split mit 3.225
Trainings- und 691 Validierungsbildern. Die primäre Auswahlmetrik ist
Macro-F1; Accuracy und Balanced Accuracy ergänzen die Einordnung.

## Zeitliche Reihenfolge der Modellwahl

1. Zunächst wurde `YOLOv11n-cls` als globale Baseline trainiert. Ein
   anschließender Validierungsvergleich mit `YOLOv11s-cls` bestätigte die
   kleinere Variante als stärkere YOLO-Baseline.
2. Danach wurde DINOv3 ViT-B/16 mit eingefrorenem Backbone und linearem Head
   bewertet.
3. DeiT-Tiny from scratch ergänzte den Vergleich als ViT-Kontrollarchitektur
   ohne externe Vortrainingsinformation.
4. Das partielle Fine-Tuning der letzten zwei DINOv3-Blöcke wurde als
   begrenzte domänenspezifische Optimierungsvariante geprüft.
5. Erst anschließend folgte die lokale Analyse manueller CVAT-Rechtecke:
   direkte Crop-Inferenz, Crop-Ablation sowie Vier- und Fünf-Klassen-
   Region-Heads.

## Globale Bildklassifikation

| Modell | Protokollierte Epochen | Bester Checkpoint | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv11n-cls | 57 | 42 | 0,95803184 | 0,94772140 | 0,95347212 |
| DINOv3 frozen + Linear Head | 50 | 26 | 0,95513748 | 0,95175834 | 0,95551804 |
| DeiT-Tiny from scratch | 92 | 68 | 0,68017366 | 0,64347740 | 0,64768028 |
| DINOv3 Partial Fine-Tuning | 14 | 4 | 0,96816208 | 0,96623869 | 0,96965035 |

### YOLO-Variantenentscheidung

| Variante | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| YOLOv11n-cls | 0,9580318379 | 0,9477213967 | 0,9534721203 |
| YOLOv11s-cls | 0,9450072359 | 0,9395723401 | 0,9437327438 |

`YOLOv11n-cls` lag in allen drei Hauptmetriken über `YOLOv11s-cls` und blieb
deshalb die YOLO-Baseline. Der beste Checkpoint der größeren Variante lag in
Epoche 30; sie wurde nicht weiter für die finale Modellmenge priorisiert.

Klassenweise Validierungsmetriken von `YOLOv11n-cls`:

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 0,964286 | 0,880435 | 0,920455 |
| Zweite Bearbeitungsstufe | 0,954315 | 0,935323 | 0,944724 |
| Fräszustand | 0,988889 | 0,988889 | 0,988889 |
| Finaler Zustand | 0,934783 | 0,986239 | 0,959821 |

Confusion Matrix, Zeilen = Referenz und Spalten = Vorhersage. Reihenfolge:
Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe, Fräszustand, Finaler
Zustand.

```text
[81,   6,   2,   3]
[ 1, 188,   0,  12]
[ 2,   0, 178,   0]
[ 0,   3,   0, 215]
```

Die häufigsten Verwechslungen waren Zweite Bearbeitungsstufe zu Finaler
Zustand mit zwölf Fällen und Erste zu Zweite Bearbeitungsstufe mit sechs
Fällen.

### DINOv3 frozen + Linear Head

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 0,977011 | 0,923913 | 0,949721 |
| Zweite Bearbeitungsstufe | 0,930000 | 0,925373 | 0,927681 |
| Fräszustand | 0,994444 | 0,994444 | 0,994444 |
| Finaler Zustand | 0,937500 | 0,963303 | 0,950226 |

Confusion Matrix, Zeilen = Referenz und Spalten = Vorhersage. Reihenfolge:
Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe, Fräszustand, Finaler
Zustand.

```text
[85,   6,   0,   1]
[ 1, 186,   1,  13]
[ 1,   0, 179,   0]
[ 0,   8,   0, 210]
```

### DINOv3-Entscheidung

Der eingefrorene DINOv3-Backbone erreichte bereits eine hohe Balanced
Accuracy und einen hohen Macro-F1. Durch das partielle Fine-Tuning stieg der
Validation Macro-F1 um rund 0,0141. Auch Accuracy und Balanced Accuracy
verbesserten sich. Damit blieb die Partial-Fine-Tuning-Variante der primäre
DINOv3-Kandidat für die finale Evaluation; die Frozen-Variante blieb als
kontrollierter Transfer-Learning-Vergleich erhalten.

### DINOv3 Partial Fine-Tuning

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 1,000000 | 0,945652 | 0,972067 |
| Zweite Bearbeitungsstufe | 0,928230 | 0,965174 | 0,946341 |
| Fräszustand | 0,994475 | 1,000000 | 0,997230 |
| Finaler Zustand | 0,971963 | 0,954128 | 0,962963 |

Confusion Matrix, Zeilen = Referenz und Spalten = Vorhersage. Reihenfolge:
Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe, Fräszustand, Finaler
Zustand.

```text
[87,   5,   0,   0]
[ 0, 194,   1,   6]
[ 0,   0, 180,   0]
[ 0,  10,   0, 208]
```

Diese Variante ist aufgrund des höchsten globalen Validation Macro-F1 die
ausgewählte DINOv3-Hauptvariante.

### DeiT-Tiny-Einordnung

DeiT-Tiny blieb deutlich hinter den vortrainierten Ansätzen zurück. Das
Ergebnis stützt die methodische Annahme, dass ein Vision Transformer ohne
Vortraining auf dem begrenzten Datensatz schwerer zu optimieren ist. Der Lauf
wird als Architekturkontrolle beibehalten und nicht mit einem nachträglichen
Wechsel auf vortrainierte Gewichte vermischt.

Klassenweise Validierungsmetriken:

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 0,476190 | 0,434783 | 0,454545 |
| Zweite Bearbeitungsstufe | 0,598291 | 0,696517 | 0,643678 |
| Fräszustand | 0,778523 | 0,644444 | 0,705167 |
| Finaler Zustand | 0,776786 | 0,798165 | 0,787330 |

Confusion Matrix, Zeilen = Referenz und Spalten = Vorhersage. Reihenfolge:
Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe, Fräszustand, Finaler
Zustand.

```text
[40,  36,   9,   7]
[19, 140,  13,  29]
[19,  31, 116,  14]
[ 6,  27,  11, 174]
```

Es sind 92 Epochen protokolliert, der beste Checkpoint stammt aus Epoche 68
und danach liegen 24 Epochen ohne weitere Verbesserung vor. Ein regulärer
Abschluss durch Early Stopping ist nicht vollständig dokumentiert und wird
nicht als nachweislich ausgelöst bezeichnet.

## Direkte DINOv3-Regionenauswertung

Die direkte lokale Auswertung verwendete den globalen
DINOv3-Partial-Fine-Tuning-Checkpoint ohne zusätzliches Training. Auf dem
Validierungssplit standen 35 Regionen der vier Schleifgradklassen für die
Metrikberechnung zur Verfügung. Sieben `Nicht_bewertbar`-Regionen wurden
separat inferiert und nicht als Fehler in der Vier-Klassen-Metrik gezählt.

| Crop-Modus | Kontextmarge | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `pad_square` | 0,00 | 0,6571428571 | 0,7106643357 | 0,6505797101 |
| `pad_square` | 0,10 | 0,6571428571 | 0,7106643357 | 0,6741666667 |
| `pad_square` | 0,25 | 0,6285714286 | 0,7014860140 | 0,6622212860 |
| `pad_square` | 0,50 | 0,6571428571 | 0,6881555944 | 0,6473930481 |
| `stretch_resize` | 0,00 | **0,6857142857** | **0,7489073427** | **0,7232600733** |

`stretch_resize` ohne Kontextmarge erreichte den höchsten Macro-F1 und wurde
deshalb als Eingabestrategie für die lokalen Region-Heads übernommen. Die
Auswahl erfolgte ausschließlich auf Validation.

Klassenweise Metriken der besten direkten Variante:

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 0,857143 | 0,461538 | 0,600000 |
| Zweite Bearbeitungsstufe | 0,526316 | 0,909091 | 0,666667 |
| Fräszustand | 1,000000 | 0,625000 | 0,769231 |
| Finaler Zustand | 0,750000 | 1,000000 | 0,857143 |

Confusion Matrix der besten Variante, Zeilen = Referenz und Spalten =
Vorhersage. Reihenfolge: Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe,
Fräszustand, Finaler Zustand.

```text
[6,  7, 0, 0]
[0, 10, 0, 1]
[1,  2, 5, 0]
[0,  0, 0, 3]
```

Die häufigste Verwechslung war Erste zu Zweite Bearbeitungsstufe mit sieben
Regionen.

## Lokale Region-Heads

| Modell | Validierungsregionen | Bester Epoch | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv3 Region-Head, vier Klassen | 35 | 10 | 0,7714285714 | 0,8116258741 | 0,7767857143 |
| DINOv3 Region-Head, fünf Klassen | 42 | 2 | 0,7142857143 | 0,7517732268 | 0,7160455487 |

### Vier-Klassen-Hauptvariante

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 0,9091 | 0,7692 | 0,8333 |
| Zweite Bearbeitungsstufe | 0,6154 | 0,7273 | 0,6667 |
| Fräszustand | 1,0000 | 0,7500 | 0,8571 |
| Finaler Zustand | 0,6000 | 1,0000 | 0,7500 |

Confusion Matrix, Zeilen = Referenz und Spalten = Vorhersage. Reihenfolge:
Erste Bearbeitungsstufe, Zweite Bearbeitungsstufe, Fräszustand, Finaler
Zustand.

```text
[10, 3, 0, 0]
[ 1, 8, 0, 2]
[ 0, 2, 6, 0]
[ 0, 0, 0, 3]
```

Die häufigsten Verwechslungen waren Erste zu Zweite Bearbeitungsstufe mit
drei Fällen, Fräszustand zu Zweite Bearbeitungsstufe mit zwei Fällen und
Zweite Bearbeitungsstufe zu Finaler Zustand mit zwei Fällen.

Der lokale Head verbesserte alle drei Hauptmetriken gegenüber der direkten
Crop-Inferenz. Er wurde daher als Hauptvariante der lokalen
Vier-Klassen-Klassifikation festgelegt. Die kleine Validierungsmenge und der
frühe beste Checkpoint zeigen zugleich ein erhebliches Overfitting-Risiko.

### Fünf-Klassen-Zusatzexperiment

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 0,9000 | 0,6923 | 0,7826 |
| Zweite Bearbeitungsstufe | 0,6154 | 0,7273 | 0,6667 |
| Fräszustand | 0,7143 | 0,6250 | 0,6667 |
| Finaler Zustand | 0,6000 | 1,0000 | 0,7500 |
| Nicht_bewertbar | 0,7143 | 0,7143 | 0,7143 |

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

Fünf von sieben `Nicht_bewertbar`-Regionen wurden korrekt erkannt. Die
Gesamtmetriken lagen unter denen des Vier-Klassen-Heads, wobei beide
Experimente wegen der unterschiedlichen Zielräume nicht als vollständig
identische Aufgaben zu interpretieren sind. Für die lokale
Schleifgradklassifikation blieb der Vier-Klassen-Head die Hauptvariante; der
Fünf-Klassen-Head dokumentiert ergänzend die praktische Ausschlussklasse.

## Trennung von Validation und Test

Keine in diesem Dokument aufgeführte Zahl ist eine finale Testmetrik. Die
Auswahl von Modellvarianten, Crop-Modus und Checkpoints war vor dem Zugriff
auf Testprädiktionen und Testmetriken abgeschlossen. Die spätere finale
Testauswertung führte zu keiner weiteren Modell- oder
Hyperparameterentscheidung.

## Verwandte Dokumente

Parameter und Laufbedingungen stehen im [experimentellen
Aufbau](experimental_setup.md). Die lokale Datengrundlage erläutert die
[Regionenanalyse](region_analysis.md). Finale Testwerte werden ausschließlich
in der [finalen Testauswertung](final_test_evaluation_result.md) geführt.
