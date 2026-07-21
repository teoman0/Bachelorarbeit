# Datensatz

## Zweck und Geltungsbereich

Dieses Dokument ist die kanonische Beschreibung des verwendeten Datensatzes,
seiner Qualitätsprüfung und der verbindlichen gruppierten Splitstrategie. Es
enthält die für Methodik und Ergebnisinterpretation maßgeblichen
Datensatzfakten, jedoch keine Rohbilder oder vertraulichen lokalen Pfade.

## Datensatzgrundlage

Die Untersuchung verwendet einen lokalen Bilddatensatz metallischer
Oberflächen zur Klassifikation von vier Bearbeitungs- beziehungsweise
Schleifzuständen. Der Datensatz wird im Projekt als `BMW_25` beziehungsweise
`Viertel BMW gefiltert` bezeichnet. Rohbilder, detaillierte Inventarlisten,
Bildhashes und Beispielausschnitte werden nicht im Repository veröffentlicht.

Der strukturelle Audit wurde mit `scripts/audit_dataset.py` durchgeführt. Er
inventarisierte die Klassenordner, prüfte die Lesbarkeit, erfasste technische
Bildmerkmale und suchte nach identischen oder per Difference-Hash ähnlichen
Bildpaaren. Alle 4.607 gefundenen Bilddateien waren lesbar; beschädigte Dateien
wurden nicht festgestellt.

## Klassen

Für die wissenschaftliche Darstellung werden folgende Klassenbezeichnungen
verwendet:

| Klasse | Bilder | Anteil |
| --- | ---: | ---: |
| Erste Bearbeitungsstufe | 614 | 13,3 % |
| Zweite Bearbeitungsstufe | 1.340 | 29,1 % |
| Fräszustand | 1.198 | 26,0 % |
| Finaler Zustand | 1.455 | 31,6 % |
| **Gesamt** | **4.607** | **100,0 %** |

Im Split-Manifest und in einzelnen lokalen Ordnernamen lautet das technische
Rohlabel der ersten Klasse `Erste Bearbeitungssufe Viertel`. Das fehlende `t`
ist ein historischer Tippfehler im Datenbestand. Er wird in technischen
Mappings beibehalten, damit Dateipfade, Checkpoints und Prediction-Tabellen
eindeutig auflösbar bleiben. In Fließtext und Ergebnistabellen wird die
korrekte Bezeichnung `Erste Bearbeitungsstufe` verwendet. Der Zusatz
`Viertel` in den technischen Labels verweist auf die Bildherkunft und ist
kein eigener Schleifgrad.

Die Verteilung ist erkennbar unausgewogen. Deshalb werden neben Accuracy auch
Balanced Accuracy, Macro-F1 und klassenweise Metriken berichtet.

## Technische Bildstruktur

| Merkmal | Befund |
| --- | --- |
| Dateiformate | 4.441 JPG, 166 BMP |
| Häufigste Auflösung | 1.824 × 1.824 Pixel bei 4.433 Bildern |
| Unterschiedliche Auflösungen | 5 |
| Breitenbereich | 960 bis 2.664 Pixel |
| Höhenbereich | 960 bis 2.304 Pixel |
| Kanalzahlen | 4.441 Bilder mit drei Kanälen, 166 mit einem Kanal |

Die Datenloader vereinheitlichen Bilder zu RGB. Abweichende Bildgrößen und
nichtquadratische Bilder werden durch das jeweils dokumentierte
modellabhängige Preprocessing behandelt; Rohbilder werden dabei weder kopiert
noch verändert.

## Viertelbilder und Gruppenbildung

Alle Dateien besitzen ein erkanntes Viertel-Suffix der Form `q1` bis `q4`
oder eine unterstützte äquivalente Schreibweise. Die vier Viertel eines
Ursprungsbildes dürfen nicht auf unterschiedliche Splits verteilt werden. Die
Gruppen-ID entsteht deshalb durch Entfernen des Suffixes aus dem Dateinamen:

```text
^(?P<group>.+?)[_-](?:q|quarter|viertel)[_-]?[1-4]$
```

Aus 4.607 Bildern wurden 1.547 Gruppen abgeleitet. Bei allen Bildern wurde ein
Viertel-Suffix erkannt; es gab keinen Fallback auf den vollständigen
Dateinamen. Die Heuristik bildet die verfügbare Originalbildstruktur ab.
Belastbare Bauteil-, Proben- oder Aufnahme-IDs hätten methodisch Vorrang,
lagen für diesen Split jedoch nicht als bessere Gruppierungsvariable vor.

Neunzehn Gruppen enthalten Bilder mit mehr als einem technischen Klassenlabel.
Diese Gruppen wurden trotzdem als unteilbare Einheiten behandelt. Der Befund
zeigt, dass Viertel derselben Ursprungsaufnahme unterschiedliche lokale
Bearbeitungszustände enthalten können und dass ein globales Bildlabel lokale
Heterogenität nur eingeschränkt beschreibt.

## Gruppierter Split

Der Split wurde vor jeder Patch- oder Regionenverarbeitung mit
`scripts/create_grouped_split.py` erzeugt. Gruppen wurden als unteilbare
Einheiten in ein möglichst klassenstratifiziertes Verhältnis von 70/15/15
aufgeteilt.

| Split | Bilder | Gruppen | Anteil |
| --- | ---: | ---: | ---: |
| Training | 3.225 | 1.083 | 70,0 % |
| Validation | 691 | 232 | 15,0 % |
| Test | 691 | 232 | 15,0 % |

| Split | Erste | Zweite | Fräszustand | Final |
| --- | ---: | ---: | ---: | ---: |
| Training | 430 | 938 | 838 | 1.019 |
| Validation | 92 | 201 | 180 | 218 |
| Test | 92 | 201 | 180 | 218 |

Die Prüfungen bestätigten:

- keine `group_id` kommt in mehreren Splits vor;
- alle 4.607 Bilder sind genau einmal zugeordnet;
- es gibt keine doppelten `image_id`- oder `relative_path`-Einträge;
- Train, Validation und Test enthalten 1.083, 232 beziehungsweise 232
  disjunkte Gruppen.

Damit wird gruppenübergreifendes Data Leakage zwischen q1- bis q4-Ableitungen
desselben Ursprungsbildes vermieden. Patches, Crops und Regionen übernehmen
stets den bereits festgelegten Split des zugehörigen Ursprungsbildes.

## Datenqualitätsgrenzen

Der Audit identifizierte zehn Paare mit identischem MD5-Hash; alle zehn lagen
klassenübergreifend vor. Zusätzlich wurden 6.947 Paare mit ähnlichem
Difference-Hash gefunden, davon 4.674 zwischen verschiedenen Klassen. Der
perzeptuelle Hash ist nur ein Suchindikator und kein Beleg für ein Duplikat.
Die identischen klassenübergreifenden Bildpaare sind dagegen eine konkrete
Datenlimitierung: Sie können auf widersprüchliche globale Labels,
Mehrfachablagen oder lokal unterschiedliche fachliche Zuordnungen hinweisen.
Diese Einschränkung ist bei der Interpretation der Modellleistung zu
berücksichtigen.

Von den zehn identischen klassenübergreifenden Paaren liegen sechs im
Trainingssplit, zwei in Validation und zwei im Testsplit. Die beiden Bilder
jedes Paars teilen jeweils Split und `group_id`. Damit entsteht durch diese
Paare kein Split-Leakage. Da identische Bildinhalte unterschiedliche
Referenzlabels besitzen, bleiben sie dennoch ein konkreter Hinweis auf
widersprüchliche globale Zielwerte.

## Manifest und Versionierung

Das verbindliche Manifest liegt unter
`data/splits/bmw25_grouped_split_manifest.csv`. Es enthält stabile Bild-IDs,
relative Pfade, technische Labels, Split-Zuordnungen, Gruppen-IDs und kleine
Bildmetadaten, aber keine absoluten lokalen Pfade oder Bildinhalte. Es bildet
die gemeinsame Referenz für Training, Validierung, finale Evaluation und die
Zuordnung manueller Regionen.

Ob das originale Manifest mit seinen relativen Dateinamen und Gruppen-IDs in
einer öffentlichen Abgabe verbleiben darf, ist eine separate Datenschutz- und
Freigabeentscheidung. Dieses Dokument nimmt diese Entscheidung nicht vorweg.
Unabhängig davon bleiben Rohbilder, Inventarlisten mit Hashes, Patches,
Feature-Caches und abgeleitete Bilddaten außerhalb des Repositorys.

## Verwandte Dokumente

Das Untersuchungsdesign ist in der [Methodik](methodology.md) beschrieben.
Technische Schritte zur Prüfung und Split-Erzeugung stehen in der
[Reproduzierbarkeitsanleitung](reproducibility.md). Ausgeführte Experimente
und ihre Ergebnisse dokumentieren der [experimentelle
Aufbau](experimental_setup.md) und die
[Validierungsergebnisse](validation_results.md).
