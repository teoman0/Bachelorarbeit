# Evaluationskonzept

Stand: 2026-07-15

Diese Datei bereitet Kapitel 3.4 "Evaluationskonzept" der Bachelorarbeit vor.
Sie beschreibt die geplante Bewertung der globalen Bildklassifikation und die
separate qualitative Einordnung patchbasierter DINOv3-Heatmaps. Es werden keine
Trainingslaeufe, keine Modellgewichte, keine Rohdaten und keine echten
Prediction-Dateien erzeugt.

## 1. Ziel der Evaluation

Das zentrale Ziel der Evaluation ist eine quantitative Bewertung der globalen
Bildklassifikation. Verglichen werden Modelle, die pro Eingabebild genau eine
von vier Schleifgradklassen vorhersagen. Die globale Auswertung soll zeigen,
welche Modellgruppe fuer die bildbasierte Schleifgradklassifikation unter einer
sauberen, gruppierten Split-Strategie am geeignetsten ist.

Davon getrennt wird die patchbasierte DINOv3-Analyse bewertet. Sie dient nur
der qualitativen Betrachtung lokaler Klassifikationskarten bzw. Heatmaps. Diese
Heatmaps koennen Hinweise darauf geben, welche Bildbereiche fuer lokale
Vorhersagen plausibel erscheinen. Sie sind jedoch keine echte semantische
Segmentierung, weil keine pixelgenauen Ground-Truth-Masken vorliegen.

## 2. Testset-Regel

Das Testset darf nur einmal fuer die finale Bewertung verwendet werden. Es wird
nicht fuer Modellwahl, Hyperparameterwahl, Patchgroessenwahl,
Heatmap-Schwellenwerte, Checkpoint-Auswahl oder qualitative Exploration
verwendet.

Alle Entscheidungen, die das Modell, die Trainingskonfiguration oder die
Auswertungslogik beeinflussen, muessen auf dem Trainings- und
Validierungssplit getroffen werden. Der Validierungssplit ist damit die
Grundlage fuer Modellentscheidungen, Hyperparameter, Checkpoint-Auswahl und
optionale Designentscheidungen. Das Testset bleibt bis zur finalen Evaluation
unangetastet.

## 3. Quantitative Metriken

Fuer die globale Bildklassifikation werden folgende Metriken berichtet:

| Metrik | Rolle in der Evaluation |
| --- | --- |
| Accuracy | Anteil korrekt klassifizierter Bilder ueber alle Testbeispiele. Sie ist leicht verstaendlich, aber bei ungleichen Klassenverteilungen nur begrenzt aussagekraeftig. |
| Balanced Accuracy | Mittelwert der klassenweisen Recall-Werte. Sie reduziert den Einfluss grosser Klassen und ist wichtig, wenn Klassen unterschiedlich haeufig vorkommen. |
| Macro-F1 | Mittelwert der F1-Werte ueber alle Klassen. Jede Klasse geht gleich stark ein, unabhaengig von ihrer Hauefigkeit. |
| Precision pro Klasse | Anteil der Vorhersagen einer Klasse, die tatsaechlich zu dieser Klasse gehoeren. Diese Metrik zeigt, ob eine Klasse zu oft vorhergesagt wird. |
| Recall pro Klasse | Anteil der tatsaechlichen Beispiele einer Klasse, die korrekt gefunden werden. Diese Metrik zeigt, ob eine Klasse uebersehen wird. |
| Confusion Matrix | Tabellarische Darstellung der Verwechslungen zwischen wahren und vorhergesagten Klassen. Sie ist Grundlage fuer die Fehleranalyse. |

Accuracy allein reicht fuer diesen Datensatz nicht aus, weil die Klassen
ungleich verteilt sind. Ein Modell kann eine hohe Gesamt-Accuracy erreichen,
indem es vor allem haeufige Klassen gut vorhersagt, waehrend seltenere Klassen
systematisch schlechter erkannt werden. Balanced Accuracy, Macro-F1 und
klassenweise Precision/Recall machen solche Unterschiede sichtbar.

## 4. Primaere und sekundaere Metriken

Als primaere Vergleichsmetrik fuer den Hauptvergleich wird Macro-F1 empfohlen.
Macro-F1 ist fuer vier Klassen mit ungleicher Klassenverteilung geeignet, weil
jede Klasse gleich gewichtet wird und Precision sowie Recall gemeinsam in die
Bewertung eingehen. Dadurch werden Modelle bestraft, die einzelne Klassen
vernachlaessigen, auch wenn die Gesamt-Accuracy hoch erscheint.

Balanced Accuracy wird als zentrale ergaenzende Metrik empfohlen. Sie ist
besonders hilfreich, um die klassenweise Trefferquote bei ungleich grossen
Klassen zu beurteilen. Accuracy bleibt als gut interpretierbare Zusatzmetrik im
Bericht erhalten, darf aber nicht allein zur Modellauswahl dienen.

## 5. Modellvergleich

Der globale Hauptvergleich umfasst:

- YOLOv11-cls;
- DINOv3 frozen + Klassifikationskopf;
- DeiT-Tiny from scratch.

Alle drei Modellgruppen werden auf demselben finalen Testsplit bewertet. Die
Voraussetzung ist, dass derselbe gruppierte Split und dieselben
Split-Manifeste verwendet werden. Die Bewertung erfolgt auf globaler Bildebene:
ein Bild, eine wahre Klasse, eine vorhergesagte Klasse.

DINOv3 patchbasiert wird nicht in denselben globalen Hauptvergleich eingeordnet.
Die patchbasierte Auswertung ist eine qualitative Zusatzanalyse fuer lokale
Klassifikationskarten. Sie darf nicht so berichtet werden, als waere sie eine
pixelgenau ueberpruefte Segmentierung.

## 6. Fehleranalyse

Die Fehleranalyse soll nach der quantitativen Auswertung folgende Punkte
betrachten:

- Confusion Matrix fuer die vier Schleifgradklassen;
- klassenweise Precision, Recall und F1-Werte;
- Sichtung typischer Fehlklassifikationen;
- besondere Beachtung von Verwechslungen zwischen benachbarten oder visuell
  aehnlichen Schleifgraden.

Vor der finalen Testauswertung sollen qualitative Fehlerbeispiele aus dem
Validierungssplit genutzt werden. Nach der finalen Testauswertung duerfen
ausgewaehlte Testfehler im Bericht beschrieben werden, sofern daraus keine
nachtraegliche Modell- oder Schwellenwertentscheidung abgeleitet wird.

## 7. Optionale ordinale Auswertung

Eine ordinale Zusatzbewertung ist optional. Sie soll nur verwendet werden, wenn
die fachliche Reihenfolge der vier Schleifgradklassen vorab eindeutig
dokumentiert ist.

Moegliche Zusatzmetriken sind:

- mittlerer absoluter Klassenabstand zwischen wahrer und vorhergesagter Klasse;
- Confusion Matrix mit ordinaler Interpretation, zum Beispiel mit besonderem
  Blick auf Verwechslungen um eine oder mehrere Klassenstufen.

Diese ordinale Auswertung ersetzt nicht Macro-F1, Balanced Accuracy oder die
klassenweise Fehleranalyse. Sie kann lediglich ergaenzen, ob Fehler fachlich
eher "nah" oder "weit" von der Zielklasse entfernt liegen.

## 8. Patchbasierte Heatmap-Evaluation

Fuer patchbasierte DINOv3-Heatmaps werden keine IoU, kein mAP, keine
Pixel-Accuracy und keine anderen pixelgenauen Segmentierungsmetriken berichtet.
Der Grund ist, dass keine pixelgenauen Masken als Ground Truth vorliegen.

Die Heatmaps werden stattdessen qualitativ plausibilisiert. Bewertet werden:

- lokale Konsistenz der Klassifikationskarten;
- Uebereinstimmung auffaelliger Heatmap-Bereiche mit sichtbaren
  Oberflaechenbereichen;
- Plausibilitaet von Uebergaengen zwischen benachbarten Bildbereichen;
- typische Faelle, in denen lokale Klassifikationshinweise stabil oder
  widerspruechlich wirken.

Die korrekte Formulierung lautet daher: qualitative lokale
Klassifikationskarten bzw. Heatmaps. Es handelt sich nicht um eine
Ground-Truth-Segmentierung und nicht um eine quantitativ validierte
Segmentationsleistung.

## 9. Reproduzierbarkeit der Evaluation

Pro Modelllauf sollen mindestens folgende Informationen gespeichert werden:

- Prediction-Datei;
- Config-ID bzw. Pfad zur verwendeten Config;
- Split-Version bzw. Split-Manifest;
- Git-Commit;
- Paketversionen;
- Modellvariante;
- Seed;
- Hardwareinformationen;
- Auswertungszeitpunkt.

Diese Informationen muessen ausreichen, um eine Auswertung spaeter auf dieselbe
Prediction-Datei, denselben Split und dieselbe Modellkonfiguration
zurueckzufuehren. Rohdaten, Modellgewichte, Checkpoints, Feature-Caches und
grosse Outputs werden nicht versioniert.

## 10. Vorschlag fuer das Prediction-Dateiformat

Fuer die globale Evaluation wird ein CSV-Schema mit einer Zeile pro Bild
empfohlen:

```text
image_id,split,true_label,predicted_label,prob_<class_1>,prob_<class_2>,prob_<class_3>,prob_<class_4>,model_name,config_id,seed
```

Pflichtfelder fuer die Metrikberechnung sind `true_label` und
`predicted_label`. `image_id` und `split` werden empfohlen, damit die
Auswertung eindeutig auf ein Bild und einen Split bezogen werden kann. Die
`prob_<class>`-Spalten sind optional, aber nuetzlich fuer spaetere Analysen wie
Konfidenzbetrachtungen oder Fehlersichtungen. Lokale absolute Rohdatenpfade
sollten nicht in Prediction-Dateien gespeichert werden.

## 11. Offene Entscheidungen

| Entscheidungspunkt | Empfehlung | Status | Begruendung |
| --- | --- | --- | --- |
| Primaere Metrik | Macro-F1 als primaere Vergleichsmetrik | empfohlen, vor finaler Evaluation festzuschreiben | Vier Klassen und ungleiche Klassenverteilung; jede Klasse soll gleich stark in den Hauptvergleich eingehen |
| Anzahl Seeds / Wiederholungen | Mindestens 3 Seeds pro Modellgruppe, falls Laufzeitbudget und Rechenressourcen reichen; sonst 1 Seed transparent begruenden | offen | Mehrere Seeds stabilisieren den Vergleich, koennen aber bei Trainingszeit und Hardware begrenzend sein |
| Ordinale Auswertung ja/nein | Optional ja, aber nur bei vorab dokumentierter Klassenreihenfolge | offen | Ohne fachlich eindeutige Reihenfolge waeren Klassenabstaende methodisch nicht belastbar |
| Laufzeit-/Ressourcenbewertung ja/nein | Ja als sekundaere Kontextinformation, nicht als primaere Leistungsmetrik | empfohlen | Trainingsdauer, Inferenzzeit und Hardwarebedarf helfen bei der praktischen Einordnung der Modelle |
| Qualitative Fehlerbeispiele | Ja, Validierungsbeispiele fuer Exploration; finale Testbeispiele nur nach abgeschlossener Modellwahl fuer den Bericht | empfohlen | Sichtbare Fehlklassifikationen erklaeren Metriken, duerfen aber das finale Testset nicht fuer Nachentscheidungen oeffnen |
| Patch-Heatmap-Bewertung | Qualitative Plausibilitaetspruefung, keine Segmentierungsmetriken | empfohlen | Es fehlen pixelgenaue Masken; Heatmaps sind lokale Klassifikationskarten und keine Ground-Truth-Segmentierung |
