# Arbeitsregeln fuer Coding Agents

Dieses Repository dient einer Bachelorarbeit zur bildbasierten Bewertung des Schleifgrades metallischer Oberflaechen. Aenderungen sollen reproduzierbar, nachvollziehbar und methodisch vorsichtig erfolgen.

## Grundsatz

Arbeite schrittweise. Implementiere keinen Trainingscode, bevor die Datensatzstruktur, Split-Strategie und Leakage-Risiken dokumentiert wurden.

## Strikte Regeln

- Keine Rohdaten, Checkpoints, Modellgewichte, lokalen Caches oder grossen Outputs ins Repository einchecken.
- Keine echten Segmentierungsmodelle trainieren, solange keine pixelgenauen Masken vorliegen.
- Patchbasierte Experimente duerfen nur auf bereits gesplitteten Originalbildern, Bauteilen oder Aufnahmen aufsetzen.
- Train/Val/Test-Splits muessen vor der Patch-Erzeugung erfolgen.
- Das Testset ist ausschliesslich fuer die finale Bewertung vorgesehen und darf nicht fuer Modellauswahl, Hyperparameter-Tuning oder Schwellenwertwahl genutzt werden.
- Alle Experimente sollen ueber Config-Dateien in `configs/` steuerbar sein.
- Jede Auswertung muss die verwendete Config, den Git-Commit, den Seed und die Paketversionen nachvollziehbar speichern.

## Erwarteter Arbeitsstil

- Bestehende Dokumentation zuerst lesen.
- Kleine, nachvollziehbare Commits bevorzugen.
- Methodische Entscheidungen in `docs/` dokumentieren, bevor sie in Code gegossen werden.
- Explorative Notebooks duerfen genutzt werden, aber produktionsrelevante Logik gehoert spaeter nach `src/` oder `scripts/`.
- Lizenzentscheidungen fuer Modelle und Gewichte in [docs/model_licenses.md](docs/model_licenses.md) pflegen.

## Bevor Trainingscode entsteht

Pruefe und dokumentiere mindestens:

- Klassen und Labeldefinitionen.
- Anzahl Bilder pro Klasse.
- Zuordnung von Bildern zu Bauteilen, Proben, Aufnahmen oder Aufnahmeserien.
- Ob mehrere Bilder oder Patches aus demselben Bauteil existieren.
- Welche Ebene fuer den Split methodisch korrekt ist: Bild, Bauteil, Probe oder Aufnahmeserie.
- Ob Split-Manifeste versioniert werden koennen, ohne sensible Daten oder grosse Dateien einzuschliessen.
