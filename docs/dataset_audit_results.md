# Ergebnisse des Datensatz-Audits

## 1. Audit-Ueberblick

- Datensatzname: BMW_25 / Viertel BMW gefiltert
- Lokaler Datenpfad: anonymisiert als BMW_25 / Viertel BMW gefiltert
- Datum/Zeit des Audits: 2026-07-13 13:19 CEST (11:19 UTC)
- Verwendetes Skript: `scripts/audit_dataset.py`
- Gesamtzahl Bilder: 4607
- Lesbare Bilder: 4607
- Korrupte/unlesbare Bilder: 0
- Anzahl Klassen: 4

## 2. Klassenverteilung

| Klasse | Anzahl Bilder | Anteil in Prozent |
| --- | ---: | ---: |
| Erste Bearbeitungsstufe Viertel | 614 | 13.3 |
| Zweite Bearbeitungsstufe Viertel | 1340 | 29.1 |
| Fraeszustand Viertel | 1198 | 26.0 |
| Finaler Zustand Viertel | 1455 | 31.6 |
| Total | 4607 | 100.0 |

Der Datensatz ist nicht extrem, aber deutlich unausgewogen. Die kleinste Klasse
ist `Erste Bearbeitungsstufe Viertel` mit 614 Bildern; die groesste Klasse ist
`Finaler Zustand Viertel` mit 1455 Bildern. Fuer spaetere Modellbewertungen
reicht Accuracy allein deshalb nicht aus. Balanced Accuracy, Macro-F1 sowie
klassenweise Precision und Recall sollten zusaetzlich berichtet werden.

## 3. Bildformate und Bildgroessen

Verwendete Dateiendungen:

| Dateiendung | Anzahl Bilder |
| --- | ---: |
| `.jpg` | 4441 |
| `.bmp` | 166 |

Die Bilder haben nicht alle dieselbe Aufloesung. Es wurden fuenf unterschiedliche
Aufloesungen gefunden. Die haeufigste Aufloesung ist `1824 x 1824` Pixel mit
4433 Bildern (96.2 Prozent). Weitere Aufloesungen treten deutlich seltener auf.

| Kennwert | Wert |
| --- | ---: |
| Minimale Breite | 960 px |
| Maximale Breite | 2664 px |
| Minimale Hoehe | 960 px |
| Maximale Hoehe | 2304 px |
| Haeufigste Aufloesung | 1824 x 1824 px |
| Unterschiedliche Aufloesungen | 5 |

Aus der Kanalzahl ergibt sich, dass 4441 Bilder drei Kanaele und 166 Bilder
einen Kanal besitzen. Eine inhaltliche Farbraumpruefung wurde damit nicht
durchgefuehrt; die Kanalzahl zeigt aber, dass spaetere Datenloader den Umgang
mit ein- und dreikanaligen Bildern explizit vereinheitlichen muessen.

Vor dem Training ist eine einheitliche Groessenanpassung bzw. ein klar
dokumentiertes Preprocessing notwendig. Die meisten Bilder sind quadratisch und
gleich gross, aber einzelne Bilder weichen deutlich ab. Das hat Auswirkungen auf
Modellinput-Groessen und spaetere Patchgroessen: Patchparameter duerfen nicht
aus dem Testset abgeleitet werden und muessen so gewaehlt werden, dass kleinere
und nicht-quadratische Bilder methodisch nachvollziehbar behandelt werden.

## 4. Duplikate und aehnliche Bilder

Die Duplikat- und Aehnlichkeitsdatei wurde nur aggregiert ausgewertet. Es werden
keine Dateinamen, relativen Pfade oder Hashes veroeffentlicht.

| Befund | Gesamt | Innerhalb derselben Klasse | Zwischen unterschiedlichen Klassen |
| --- | ---: | ---: | ---: |
| Identische MD5-Duplikatpaare | 10 | 0 | 10 |
| Perceptual-hash-aehnliche Paare | 6947 | 2273 | 4674 |

Die Duplicate-Liste wurde nicht gekappt (`duplicate_rows_truncated = false`).

Die identischen MD5-Duplikatpaare liegen ausschliesslich zwischen
unterschiedlichen Klassen. Das ist ein kritischer Hinweis auf ein moegliches
Label- oder Datenqualitaetsproblem und muss vor der Definition der Splits
manuell geprueft werden.

Zusaetzlich wurden viele perceptual-hash-aehnliche Paare gefunden, darunter ein
grosser Anteil zwischen unterschiedlichen Klassen. Da der verwendete
perceptual hash nur ein einfacher Aehnlichkeitsindikator ist, beweist dies
nicht automatisch fehlerhafte Labels. Fuer die Leakage- und Split-Diskussion ist
der Befund trotzdem wichtig: visuell verwandte, wiederholte oder aus demselben
Ursprung stammende Bilder duerfen spaeter nicht ueber verschiedene Splits
verteilt werden.

## 5. Bedeutung von "Viertel" / moegliche Gruppenstruktur

Die Ordner- und Dateinamen deuten darauf hin, dass Bilder als Viertelbilder
organisiert sind. In der Inventarliste wurde fuer alle Bilder ein Suffixmuster
der Form `q1` bis `q4` erkannt. Anonymisiert laesst sich das Muster etwa als
`aufnahmeID_q1.jpg`, `aufnahmeID_q2.jpg`, `aufnahmeID_q3.jpg`,
`aufnahmeID_q4.jpg` beschreiben.

Eine plausible automatische Gruppen-ID kann aus dem Dateinamen abgeleitet
werden, indem das Viertel-Suffix entfernt wird. Eine geeignete Regex waere:

```text
^(?P<group>.+?)[_-](?:q|quarter|viertel)[_-]?[1-4]$
```

Diese Regex sollte vor der finalen Split-Erzeugung manuell validiert werden.
Insbesondere muss geprueft werden, ob die abgeleitete Gruppe tatsaechlich einem
Originalbild, einer Aufnahme, einem Bauteil oder einer Aufnahmeserie entspricht.
Falls Bauteil- oder Proben-IDs separat existieren, haben diese fuer die
Split-Strategie Vorrang vor einer reinen Dateinamenheuristik.

## 6. Konsequenz fuer Split-Strategie

Empfohlene Prioritaet:

1. Split auf Bauteil- oder Probenebene, falls solche IDs existieren.
2. Split auf Aufnahme- oder Originalbildebene, falls Viertelbilder aus einem
   Ursprungsbild stammen.
3. Nur wenn keine belastbare Gruppierung ableitbar ist: bildbasierter Split,
   aber mit klar dokumentierter Einschraenkung.

Splits muessen vor jeder Patch-Erzeugung definiert werden. Patches desselben
Originalbildes oder derselben Gruppe duerfen nicht in unterschiedliche Splits
gelangen. Das Testset darf nicht fuer Modellauswahl, Hyperparameterwahl,
Patchgroesse, Heatmap-Schwellenwerte oder qualitative Exploration genutzt
werden.

## 7. Konsequenz fuer patchbasierte lokale Schleifgradvorhersage

Fuer den Datensatz liegen keine pixelgenauen Segmentierungsannotationen vor.
Deshalb wird keine echte semantische Segmentierung trainiert. Patch-Labels
wuerden aus globalen Bildlabels abgeleitet und koennen lokales Labelrauschen
enthalten, insbesondere wenn einzelne Bilder heterogene Oberflaechenbereiche
zeigen.

Patchbasierte Heatmaps sind daher als qualitative lokale Klassifikationskarten
zu interpretieren, nicht als Ground-Truth-Segmentierungen. Diese Einschraenkung
muss in der Bachelorarbeit ausdruecklich genannt werden.

## 8. Textbaustein fuer Kapitel 3.1

Vor der Implementierung von Trainingscode wurde der Datensatz strukturell
geprueft. Der lokale Audit umfasst 4607 Bilddateien in vier Klassen; alle
Dateien waren lesbar, korrupte oder unlesbare Bilder wurden nicht gefunden. Die
Klassenverteilung ist ungleichmaessig, da die kleinste Klasse 614 Bilder und die
groesste Klasse 1455 Bilder enthaelt. Fuer spaetere Auswertungen sollten daher
neben Accuracy auch klassengewichtete bzw. klassenbalancierte Metriken wie
Balanced Accuracy und Macro-F1 berichtet werden.

Die Duplikat- und Aehnlichkeitspruefung ergab 10 identische MD5-Duplikatpaare
sowie 6947 Paare mit aehnlichem perceptual hash. Insbesondere identische oder
visuell sehr aehnliche Bilder zwischen unterschiedlichen Klassen sind ein
wichtiger Hinweis auf moegliche Datenqualitaets- und Leakage-Risiken. Daraus
folgt, dass Train/Validation/Test-Splits vor jeder Patch-Erzeugung und
moeglichst auf Bauteil-, Aufnahme- oder Originalbildebene definiert werden
muessen. Patchbasierte Heatmaps sind mangels pixelgenauer Annotationen nur als
qualitative lokale Klassifikationskarten und nicht als echte
Ground-Truth-Segmentierungen zu interpretieren.

## 9. Datenschutznotiz

Detaillierte Inventarlisten, lokale Bildpfade, Hashes, Duplicate-Pair-Dateien
und Sample-Grids mit echten Bildausschnitten bleiben lokal und werden nicht in
das oeffentliche Repository committed.
