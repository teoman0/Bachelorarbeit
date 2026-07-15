# Datensatz-Split-Summary

Erzeugt am: `2026-07-15T08:58:03.814578+00:00`

Diese Summary beschreibt den gruppierten Train/Val/Test-Split. Es wurden keine Bilddaten kopiert, keine Patches erzeugt und keine Modelle trainiert.

## Datensatz

- Gesamtzahl Bilder: 4607
- Anzahl Klassen: 4
- Anzahl Gruppen: 1547
- Seed: 42
- Split-Verhaeltnis: train=0.7, val=0.15, test=0.15

| Klasse | Gesamt |
| --- | --- |
| Erste Bearbeitungssufe Viertel | 614 |
| Finaler Zustand Viertel | 1455 |
| Fräszustand Viertel | 1198 |
| Zweite Bearbeitungsstufe Viertel | 1340 |

## Gruppierungslogik

- Regex: `^(?P<group>.+?)[_-](?:q|quarter|viertel)[_-]?[1-4]$`
- Logik: Remove q1-q4, quarter, or viertel suffix from filename stem; fall back to full stem.
- Bilder mit erkanntem q1-q4-/Viertel-Suffix: 4607
- Bilder ohne erkanntes q1-q4-/Viertel-Suffix: 0
- Gruppen mit mehr als einem Label: 19

## Split-Verteilung

| Split | Bilder | Gruppen | Anteil |
| --- | --- | --- | --- |
| train | 3225 | 1083 | 70.0% |
| val | 691 | 232 | 15.0% |
| test | 691 | 232 | 15.0% |

## Klassenverteilung pro Split

| Split | Klasse | Bilder |
| --- | --- | --- |
| train | Erste Bearbeitungssufe Viertel | 430 |
| train | Finaler Zustand Viertel | 1019 |
| train | Fräszustand Viertel | 838 |
| train | Zweite Bearbeitungsstufe Viertel | 938 |
| val | Erste Bearbeitungssufe Viertel | 92 |
| val | Finaler Zustand Viertel | 218 |
| val | Fräszustand Viertel | 180 |
| val | Zweite Bearbeitungsstufe Viertel | 201 |
| test | Erste Bearbeitungssufe Viertel | 92 |
| test | Finaler Zustand Viertel | 218 |
| test | Fräszustand Viertel | 180 |
| test | Zweite Bearbeitungsstufe Viertel | 201 |

## Pruefungen

| Pruefung | Ergebnis |
| --- | --- |
| Keine group_id in mehreren Splits | True |
| Alle Bilder genau einmal zugeordnet | True |
| Leaking group count | 0 |
| Duplicate image_id count | 0 |
| Duplicate relative_path count | 0 |

## Versionierung

Das Manifest enthaelt relative Pfade zum lokalen Dataset-Root. Vor einem Commit muss geprueft werden, ob diese relativen Dateinamen datenschutzrechtlich versionierbar sind.
