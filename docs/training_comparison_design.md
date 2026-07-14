# Trainings- und Vergleichsdesign

Stand: 2026-07-14

Diese Datei bereitet Kapitel 3.3 "Trainings- und Vergleichsdesign" vor. Sie
beschreibt die geplante Split-, Preprocessing-, Augmentierungs-, Trainings- und
Auswertungslogik, ohne finale Trainingslaeufe zu starten oder konkrete
Trainingsparameter als endgueltig festzuschreiben.

## Geltungsbereich

Der Hauptvergleich erfolgt auf globaler Bildebene fuer vier Schleifgradklassen.
Verglichen werden YOLOv11-cls, DINOv3 frozen + Klassifikationskopf und
DeiT-Tiny from scratch. DINOv3 patchbasiert ist eine qualitative Zusatzanalyse
zur lokalen Visualisierung und wird nicht als echte semantische Segmentierung
bewertet.

Keine Rohdaten, Checkpoints, Modellgewichte, Feature-Caches oder grosse Outputs
werden versioniert. Alle finalen Experimente muessen vorab ueber Config-Dateien
in `configs/` festgelegt werden.

## 1. Split-Strategie

Der Datensatz umfasst laut Audit 4607 lesbare Bilder in vier Klassen. Die
Ordner- und Dateinamen deuten auf geviertelte Oberflaechenbilder mit
`q1`- bis `q4`-Suffixen hin. Damit besteht ein Data-Leakage-Risiko, wenn Viertel
desselben Ursprungsbildes auf unterschiedliche Splits verteilt werden.

Die Gruppierungslogik soll deshalb vor dem finalen Split auf Basis der
Dateinamen validiert werden. Eine plausible `group_id` kann aus dem Dateinamen
abgeleitet werden, indem das Viertel-Suffix entfernt wird:

```text
^(?P<group>.+?)[_-](?:q|quarter|viertel)[_-]?[1-4]$
```

Beispiel:

```text
aufnahme123_q1.jpg -> group_id = aufnahme123
aufnahme123_q2.jpg -> group_id = aufnahme123
aufnahme123_q3.jpg -> group_id = aufnahme123
aufnahme123_q4.jpg -> group_id = aufnahme123
```

Empfohlen wird ein gruppierter, nach Klassen moeglichst stratifizierter
Train/Validation/Test-Split im Verhaeltnis `70/15/15`. Bei 4607 Bildern bleiben
so rund 3225 Bilder fuer Training und jeweils rund 691 Bilder fuer Validierung
und Test. Da die kleinste Klasse 614 Bilder enthaelt, liefert `70/15/15` mehr
Validierungs- und Testbeispiele pro Klasse als `80/10/10`, ohne den
Trainingsanteil zu stark zu reduzieren. Das ist besonders wichtig, weil
Balanced Accuracy, Macro-F1 und klassenweise Precision/Recall berichtet werden
sollen.

Der Split muss vor jeder Patch-Erzeugung erfolgen. Alle Viertel, Patches oder
sonstigen Ableitungen desselben Ursprungsbildes, Bauteils oder derselben
Aufnahmeserie muessen im selben Split bleiben. Falls spaeter verlaessliche
Bauteil-, Proben- oder Aufnahmeserien-IDs verfuegbar sind, haben diese Vorrang
vor der reinen `q1`-bis-`q4`-Dateinamenheuristik.

Noch wird kein finales Split-Manifest erzeugt. Spaeter soll der Split wie folgt
erstellt werden:

1. Bildinventar aus lokalen Rohdaten erzeugen, ohne Bilddaten zu kopieren.
2. Klasse, relativen Pfad, Dateiendung und abgeleitete `group_id` speichern.
3. `group_id`-Ableitung manuell anhand von Stichproben validieren.
4. Gruppenweise und moeglichst klassenstratifiziert in `train`, `val` und
   `test` aufteilen.
5. Pruefen, dass keine `group_id` in mehr als einem Split vorkommt.
6. Kleine Split-Manifestdatei versionieren, falls sie keine sensiblen Pfade
   oder Rohdaten enthaelt.

Ein versionierbares Manifest soll nur Metadaten enthalten, zum Beispiel:

```text
image_id,relative_path,label,split,group_id,width,height,channels,source_suffix
```

## 2. Bildgroesse und Preprocessing

Rohdaten werden nicht veraendert. Groessenanpassung, Kanalvereinheitlichung und
Normalisierung erfolgen im Dataset, Loader oder in nicht versionierten
temporaren Outputs.

Empfohlene Zielaufloesungen:

| Modellgruppe | Empfohlene Startgroesse | Begruendung |
| --- | ---: | --- |
| YOLOv11-cls | `224x224` als Startpunkt, optional `320x320` bei ausreichender Hardware | YOLOv11-cls kann eine eigene `imgsz`-Konfiguration nutzen; `224` erleichtert den Vergleich mit ViT-Modellen, `320` kann feine Oberflaechenstrukturen besser erhalten |
| DINOv3 frozen + Kopf | `224x224` als Startpunkt | kompatibel mit typischen ViT-Backbones und effizient fuer Feature-Extraktion; groessere ViT-kompatible Groessen bleiben eine optionale spaetere Validierungsfrage |
| DeiT-Tiny from scratch | `224x224` | passend zu `deit_tiny_patch16_224` |
| DINOv3 patchbasiert | Patchgroesse `224x224` als Startpunkt | entspricht ViT-kompatiblen Patches und bestehenden DINOv3-Prototypen |

Da die meisten Bilder `1824x1824` Pixel haben, aber einzelne Bilder kleinere,
groessere oder nicht-quadratische Aufloesungen besitzen, muss die
Resize-Strategie explizit dokumentiert werden. Fuer den ersten globalen
Vergleich wird eine deterministische Umwandlung auf eine quadratische
Zielgroesse empfohlen. Um unnoetige Verzerrung zu vermeiden, ist
`resize-with-padding` bzw. ein modellkompatibles Letterboxing methodisch
vorsichtiger als ein ungeprueftes Strecken nicht-quadratischer Bilder.

Ein- und dreikanalige Bilder werden im Loader einheitlich als RGB behandelt.
Graustufenbilder werden dabei durch Kanalduplikation in drei Kanaele
ueberfuehrt. Diese Regel verhindert, dass unterschiedliche Kanalzahlen als
unbeabsichtigtes Klassensignal in das Training eingehen.

Normalisierung soll modellabhaengig erfolgen:

- YOLOv11-cls: Ultralytics-Preprocessing und `imgsz` pro Experiment
  dokumentieren.
- DINOv3 frozen: die zur Modellquelle passende Normalisierung bzw. der passende
  Image Processor ist zu verwenden und zu dokumentieren.
- DeiT-Tiny from scratch: bevorzugt einfache, train-set-basierte oder fest
  definierte Normalisierung; Statistiken duerfen nur aus dem Trainingssplit
  geschaetzt werden.

## 3. Augmentierungen

Augmentierungen sollen vorsichtig eingesetzt werden, weil metallische
Schleifstrukturen richtungsabhaengige und feine Texturmerkmale enthalten
koennen.

Als sinnvolle Startmenge gelten:

- horizontale und vertikale Flips, falls die Klassenlabel rotations- bzw.
  spiegelungsinvariant sind;
- 90-Grad-Rotationen, falls die Orientierung des Bauteils kein Labelbestandteil
  ist;
- leichte Helligkeits- und Kontrastaenderungen zur Robustheit gegen
  Beleuchtungsschwankungen;
- leichte Crops oder Resize-Strategien, solange die sichtbare Oberflaeche nicht
  systematisch verfaelscht wird.

Zu vermeiden oder nur nach manueller Plausibilitaetspruefung einzusetzen sind
starke Perspektivtransformationen, starke Blur-Filter, aggressive
Farbveraenderungen, CutMix/MixUp ohne methodische Begruendung und zufaellige
Transformationen, die Schleifrichtung, Riefenstruktur oder lokale Defekte
unrealistisch veraendern koennen.

Fuer die Vergleichbarkeit soll die Grundlogik der Augmentierung ueber die
globalen Hauptmodelle hinweg moeglichst gleich sein. Modellabhaengige
Unterschiede sind erlaubt, wenn die jeweilige Bibliothek eigene
Standardmechanismen besitzt, muessen aber in der Config dokumentiert werden.
Validation und Test erhalten keine zufaelligen Augmentierungen, sondern nur das
deterministische Preprocessing.

## 4. Trainingsdesign nach Modellgruppe

### YOLOv11-cls

YOLOv11-cls wird als supervised Klassifikationsmodell auf globalen Bildlabels
genutzt. Die Aufgabe lautet: Bild -> eine von vier Schleifgradklassen. Es wird
nicht als Segmentierungsmodell verwendet.

Die konkrete Variante bleibt bis vor dem Training offen. Bevorzugt werden kleine
Varianten wie `yolo11n-cls` oder `yolo11s-cls`, abhaengig von Hardware,
Laufzeitbudget und Stabilitaet. Pro Experiment muessen Ultralytics-Version,
Modellvariante, Input Size, Seed, Augmentierungen, Pretraining und
Lizenzhinweise dokumentiert werden.

### DINOv3 frozen + Klassifikationskopf

DINOv3 frozen + Klassifikationskopf ist der Hauptansatz. Der DINOv3-Backbone
wird zunaechst eingefroren. Trainiert wird nur ein linearer oder kleiner
MLP-Klassifikationskopf.

Die konkrete DINOv3-Backbone-Variante ist noch vor dem Training festzulegen.
Feature-Caching ist sinnvoll, weil ein eingefrorener Backbone wiederholte
Feature-Extraktion vermeidet. Ein solcher Cache muss lokal ausserhalb von Git
liegen und die verwendete Config, Split-Version, Modellvariante und
Paketversionen referenzieren. Fine-Tuning des Backbones bleibt optional und darf
nur nach einer sauberen Validierungsbegruendung erfolgen.

### DeiT-Tiny from scratch

DeiT-Tiny from scratch dient als ViT-Kontrollmodell. Bevorzugt wird
`deit_tiny_patch16_224` mit `pretrained=False`. Dadurch werden keine externen
Gewichte genutzt, und separate Gewichts-Lizenzfragen werden vermieden.

Im Gegensatz zum DINOv3-Ansatz wird das komplette Modell trainiert. Die Rolle
dieses Modells ist eine Architekturkontrolle bzw. ViT-from-scratch-Untergrenze,
nicht ein erwarteter Leistungsfavorit gegenueber DINOv3.

### DINOv3 patchbasiert

DINOv3 patchbasiert ist nur eine Zusatzanalyse. Patches werden erst nach dem
gruppierten Split erzeugt. Patchlabels werden aus dem globalen Bildlabel
abgeleitet und koennen lokal verrauscht sein, besonders wenn ein Bild heterogene
Oberflaechenbereiche enthaelt.

Die Heatmaps werden qualitativ als lokale Klassifikationskarten interpretiert,
nicht als Ground-Truth-Segmentierung. Ohne pixelgenaue Masken wird kein echtes
semantisches Segmentierungsmodell trainiert.

## 5. Metriken

Fuer die globale Klassifikation werden folgende Metriken empfohlen:

- Accuracy;
- Balanced Accuracy;
- Macro-F1;
- Precision pro Klasse;
- Recall pro Klasse;
- Confusion Matrix.

Accuracy allein reicht nicht aus, weil die Klassenverteilung unausgewogen ist.
Die kleinste Klasse enthaelt 614 Bilder, die groesste 1455 Bilder. Ein Modell
koennte daher mit guter Gesamt-Accuracy trotzdem schwache Leistung auf der
kleinsten Klasse zeigen. Balanced Accuracy und Macro-F1 gewichten Klassen
staerker gleich und sind fuer diesen Datensatz aussagekraeftiger.

Ordinale Zusatzmetriken koennen sinnvoll sein, falls die vier Schleifgrade eine
belastbare natuerliche Reihenfolge besitzen. Moegliche Zusatzmetriken waeren
mittlerer absoluter Klassenabstand, quadratisch gewichtetes Kappa oder eine
Konfusionsmatrix mit ordinaler Interpretation. Diese Auswertung bleibt optional,
bis die Klassenreihenfolge fachlich eindeutig dokumentiert ist.

## 6. Reproduzierbarkeit

Alle Experimente folgen einem Config-First-Prinzip. Jede Modellgruppe und jeder
Lauf wird ueber eine versionierte Config in `configs/` beschrieben, bevor
Training oder Auswertung startet.

Eine finale Experiment-Config soll mindestens enthalten:

- Experimentname;
- Dataset-ID;
- Split-Manifest;
- Modellgruppe und Modellvariante;
- Input Size;
- Preprocessing;
- Augmentierungen;
- Optimizer;
- Scheduler;
- Batch Size;
- Epochen;
- Seed;
- Paketversionen;
- Lizenz- und Gewichtsquelle;
- Output-Verzeichnis.

Zusaetzlich sollen pro Lauf Git-Commit, verwendete Config, Split-Version,
Paketversionen und relevante Hardwareinformationen in einem lokalen Summary
gespeichert werden. Checkpoints, Modellgewichte, Rohdaten, Feature-Caches und
grosse Outputs werden nicht committed.

## 7. Vorschlag fuer Config-Templates

Die Dateien unter `configs/templates/` sind Platzhalter und keine finalen
Trainingsconfigs:

- `configs/templates/yolo11_cls_template.yaml`
- `configs/templates/dinov3_frozen_head_template.yaml`
- `configs/templates/deit_tiny_from_scratch_template.yaml`
- `configs/templates/dinov3_patch_analysis_template.yaml`

Sie enthalten keine echten lokalen Datenpfade. Vor einem Training muessen sie
in konkrete Experiment-Configs ueberfuehrt und mit realen, aber
versionierbaren Metadaten gefuellt werden.

## 8. Offene Entscheidungen

| Entscheidungspunkt | Empfehlung | Status | Begruendung |
| --- | --- | --- | --- |
| Split-Verhaeltnis | `70/15/15` | final fuer Startdesign | Liefert ausreichend Trainingdaten und robustere Val/Test-Groessen als `80/10/10`, besonders fuer die kleinste Klasse |
| Gruppierung ueber `q1`-`q4` | `group_id` durch Entfernen des Viertel-Suffixes ableiten | vor Training festzulegen | Regex muss an echten Dateinamen validiert werden; hoehere Bauteil-/Proben-IDs haetten Vorrang |
| Split-Zeitpunkt | Split vor Patch-Erzeugung | final | Verhindert Leakage zwischen Patches oder Vierteln desselben Ursprungsbildes |
| YOLOv11 Input Size | `224x224` als Startpunkt, optional `320x320` pruefen | vor Training festzulegen | `224` verbessert Vergleichbarkeit, `320` kann feinere Strukturen erhalten |
| DINOv3 globale Input Size | `224x224` als Startpunkt | vor Training festzulegen | Effizient und ViT-kompatibel; groessere ViT-kompatible Groessen koennen spaeter validiert werden |
| DeiT-Tiny Input Size | `224x224` | final fuer Startdesign | Passend zu `deit_tiny_patch16_224` |
| Patchgroesse und Stride | Patchgroesse `224`, Stride zunaechst `224`; Ueberlappung optional | vor Training festzulegen | Nicht-ueberlappende Patches sind einfacher interpretierbar; Ueberlappung erhoeht Aufwand und Abhaengigkeit |
| Augmentierungsstrategie | vorsichtige Flips/90-Grad-Rotationen und leichte Helligkeits-/Kontrastaenderungen | vor Training festzulegen | Schleifstrukturen duerfen nicht unrealistisch veraendert werden |
| Umgang mit Graustufenbildern | im Loader nach RGB konvertieren durch Kanalduplikation | final fuer Startdesign | Einheitliche Eingabeform verhindert Kanalzahl als Stoersignal |
| Metriken | Accuracy, Balanced Accuracy, Macro-F1, Precision/Recall je Klasse, Confusion Matrix | final fuer Startdesign | Klassen sind unausgewogen, daher reicht Accuracy allein nicht |
| Ordinale Zusatzmetriken | optional, wenn Reihenfolge fachlich belastbar ist | optional | Schleifgrade koennen ordinal sein, aber die Reihenfolge muss vorher dokumentiert werden |
| DINOv3 Feature-Caching | lokal ausserhalb von Git nutzen, falls mehrere Head-Experimente geplant sind | optional | Spart Rechenzeit bei frozen Backbone, erzeugt aber nicht versionierte Zwischenartefakte |
| DINOv3 Fine-Tuning | nur nach Validierungsbegruendung | optional | Erhoeht Aufwand und Overfitting-Risiko; Testset darf nicht zur Entscheidung genutzt werden |
| DeiT-Tiny Stabilitaet | nach ersten Val-Lernkurven beurteilen | vor Training festzulegen | ViT from scratch kann bei 4607 Bildern datenhungrig und instabil sein |
