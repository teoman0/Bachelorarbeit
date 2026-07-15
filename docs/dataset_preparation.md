# Datensatzvorbereitung

Stand: 2026-07-15

Diese Datei bereitet das Kapitel "Durchfuehrung - Datensatz" der
Bachelorarbeit vor. Sie beschreibt die konkrete Vorbereitung des Datensatzes
fuer globale Bildklassifikation und spaetere patchbasierte Zusatzanalysen. Es
werden keine Trainingslaeufe gestartet, keine Modellgewichte geladen und keine
Bilddaten versioniert.

## 1. Datensatzgrundlage

Die Grundlage ist der lokale Viertelbild-Datensatz, der im Datensatz-Audit
dokumentiert wurde. Laut Audit umfasst der Datensatz 4607 lesbare Bilder in
vier Klassen. Korrupte oder unlesbare Bilder wurden im Audit nicht gefunden.

Rohdaten bleiben lokal und liegen nicht im Repository. Das Repository enthaelt
nur Skripte, methodische Dokumentation und kleine, pruefbare Metadaten, sofern
sie keine sensiblen lokalen Pfade oder Bildinhalte enthalten.

## 2. Klassenstruktur

Die Klassen werden technisch aus den Ordnernamen unterhalb des lokalen
Dataset-Roots abgeleitet. Erwartet wird eine Struktur mit einem Unterordner pro
Schleifgradklasse:

```text
data/raw/
  <klasse_1>/
  <klasse_2>/
  <klasse_3>/
  <klasse_4>/
```

Das Audit weist vier Klassen aus. Die Klassenverteilung ist ungleich, daher
muessen spaetere Auswertungen neben Accuracy auch Balanced Accuracy, Macro-F1
und klassenweise Precision/Recall berichten.

Die echte Klassenverteilung im erzeugten Split-Manifest lautet:

| Klasse | Bilder |
| --- | ---: |
| Erste Bearbeitungssufe Viertel | 614 |
| Finaler Zustand Viertel | 1455 |
| Fräszustand Viertel | 1198 |
| Zweite Bearbeitungsstufe Viertel | 1340 |
| Gesamt | 4607 |

## 3. Technische Bildstruktur

Das Audit beschreibt 4607 lesbare Bilder mit den Dateiendungen `.jpg` und
`.bmp`. Die meisten Bilder besitzen die Aufloesung `1824 x 1824` Pixel, einzelne
Bilder weichen jedoch in Breite, Hoehe oder Kanalzahl ab. Es liegen sowohl
dreikanalige als auch einkanalige Bilder vor.

Fuer die Split-Erstellung werden Bilddateien nur gelesen, um Metadaten zu
erfassen:

- `image_id`;
- relativer Pfad zum Dataset-Root;
- Klasse;
- Dateiendung;
- Breite und Hoehe;
- Kanalzahl;
- abgeleitete `group_id`;
- zugewiesener Split.

Bilddaten werden nicht kopiert und nicht veraendert.

## 4. Gruppierungslogik q1-q4

Viele Dateien sind geviertelte Ursprungsaufnahmen mit Suffixen der Form `q1`
bis `q4` oder verwandten Schreibweisen wie `quarter` bzw. `viertel`. Damit
besteht ein Leakage-Risiko, wenn Viertel desselben Ursprungsbildes zufaellig
auf Train, Validation und Test verteilt werden.

Die automatische Gruppen-ID wird deshalb aus dem Dateinamen abgeleitet, indem
ein Viertel-Suffix entfernt wird. Die voreingestellte Regex lautet:

```text
^(?P<group>.+?)[_-](?:q|quarter|viertel)[_-]?[1-4]$
```

Beispiele:

```text
aufnahme123_q1.jpg -> group_id = aufnahme123
aufnahme123_q2.jpg -> group_id = aufnahme123
aufnahme123_q3.jpg -> group_id = aufnahme123
aufnahme123_q4.jpg -> group_id = aufnahme123
```

Wenn kein solches Suffix erkannt wird, nutzt das Skript den vollstaendigen
Dateinamen ohne Endung als `group_id`. Die Summary weist aus, wie viele Bilder
ohne erkanntes q1-q4-/Viertel-Suffix verarbeitet wurden. Falls spaeter bessere
Bauteil-, Proben- oder Aufnahme-IDs verfuegbar sind, haben diese methodisch
Vorrang vor der Dateinamenheuristik.

Im echten Splitlauf wurde fuer alle 4607 Bilder ein q1-q4-/Viertel-Suffix
erkannt. Es gab damit 0 Bilder ohne erkannte q1-q4-Gruppierung. Aus den
Dateinamen wurden 1547 Gruppen abgeleitet. 19 Gruppen enthalten Bilder aus mehr
als einem Label; diese Gruppen wurden trotzdem als unteilbare Einheiten
behandelt und jeweils nur einem Split zugeordnet.

## 5. Split-Erstellung

Das Skript [scripts/create_grouped_split.py](../scripts/create_grouped_split.py)
erstellt einen gruppierten Train/Val/Test-Split im Verhaeltnis `70/15/15`.
Gruppen werden als unteilbare Einheiten behandelt. Das Skript versucht
zusaetzlich, die Klassenverteilung pro Split moeglichst nah an der globalen
Klassenverteilung zu halten.

Ausgefuehrter lokaler Aufruf auf dem echten Datensatz:

```powershell
python scripts/create_grouped_split.py `
  --data-root <lokaler_dataset_root> `
  --output-manifest data/splits/bmw25_grouped_split_manifest.csv `
  --summary-json data/splits/bmw25_grouped_split_summary.json `
  --summary-md docs/dataset_split_summary.md `
  --seed 42
```

Falls `python` unter Windows nicht im PATH liegt, kann die lokale virtuelle
Umgebung oder der gebuendelte Python-Interpreter verwendet werden.

Der erzeugte Split umfasst:

| Split | Bilder | Gruppen | Anteil |
| --- | ---: | ---: | ---: |
| train | 3225 | 1083 | 70.0% |
| val | 691 | 232 | 15.0% |
| test | 691 | 232 | 15.0% |

Die Klassenverteilung pro Split lautet:

| Split | Erste Bearbeitungssufe Viertel | Finaler Zustand Viertel | Fräszustand Viertel | Zweite Bearbeitungsstufe Viertel |
| --- | ---: | ---: | ---: | ---: |
| train | 430 | 1019 | 838 | 938 |
| val | 92 | 218 | 180 | 201 |
| test | 92 | 218 | 180 | 201 |

## 6. Leakage-Pruefung

Nach der Split-Erstellung prueft das Skript:

- ob jede `group_id` nur in genau einem Split vorkommt;
- ob alle gelesenen Bilder genau einmal im Manifest enthalten sind;
- ob doppelte `image_id`- oder `relative_path`-Eintraege auftreten;
- wie viele Gruppen mehr als ein Label enthalten;
- wie viele Bilder ohne erkanntes q1-q4-/Viertel-Suffix verarbeitet wurden.

Die Pruefung "keine `group_id` in mehreren Splits" ist zwingend. Falls sie
fehlschlaegt, darf das Manifest nicht fuer Training oder Evaluation verwendet
werden.

Im echten Splitlauf waren alle Pruefungen erfolgreich:

| Pruefung | Ergebnis |
| --- | --- |
| Keine `group_id` in mehreren Splits | bestanden |
| Leaking groups | 0 |
| Alle Bilder genau einmal zugeordnet | bestanden |
| Manifest-Zeilen | 4607 |
| Doppelte `image_id` | 0 |
| Doppelte `relative_path` | 0 |

## 7. Preprocessing-Vorbereitung

Der Split wird vor jeder Patch-Erzeugung definiert. Patchdaten duerfen spaeter
nur innerhalb der bereits festgelegten Splits erzeugt werden. Dadurch bleiben
Patches, Viertelbilder oder sonstige Ableitungen desselben Ursprungsbildes im
gleichen Split.

Fuer spaetere Modellinputs muss das Preprocessing separat in Config-Dateien
dokumentiert werden. Dazu gehoeren insbesondere:

- einheitliche Zielgroesse;
- Umgang mit nicht-quadratischen Bildern;
- Umgang mit ein- und dreikanaligen Bildern;
- Normalisierung;
- deterministisches Validation- und Test-Preprocessing;
- keine Ableitung von Patchgroesse oder Preprocessing-Entscheidungen aus dem
  Testset.

## 8. Versionierte und lokale Dateien

Versioniert werden:

- [scripts/create_grouped_split.py](../scripts/create_grouped_split.py);
- diese Dokumentation;
- die kleine Split-Summary `docs/dataset_split_summary.md`;
- das Split-Manifest `data/splits/bmw25_grouped_split_manifest.csv`;
- die maschinenlesbare Summary
  `data/splits/bmw25_grouped_split_summary.json`.

Das Manifest wurde vor dem Commit geprueft. Es enthaelt relative Pfade,
Klassenlabels, Split-Zuordnungen, `group_id`-Werte und technische Bildmetadaten,
aber keine absoluten lokalen Pfade, keine Bilddaten, keine Hashes, keine
Modellartefakte und keine Checkpoints. Die relativen Dateinamen und
`group_id`-Werte wurden fuer diesen Arbeitsstand als unkritisch versionierbar
eingestuft.

Lokal bleiben:

- Rohbilder unter `data/raw/`;
- abgeleitete Bilddaten unter `data/interim/`, `data/processed/` und
  `data/patches/`;
- Feature-Caches unter `data/cache/`;
- Outputs, Runs, Checkpoints und Modellgewichte;
- Split-Manifeste, falls relative Dateinamen oder Gruppen-IDs als sensibel
  eingestuft werden.

Absolute lokale Pfade duerfen nicht in versionierbaren Dateien stehen. Der
lokale Dataset-Root wird deshalb nur im Arbeitskontext und nicht in den
versionierten Split-Dateien dokumentiert.
