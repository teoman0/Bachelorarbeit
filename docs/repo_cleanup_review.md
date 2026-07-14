# Repository-Konsistenzpruefung

Stand: 2026-07-14

Diese Datei dokumentiert den Konsistenz- und Aufraeum-Check vor Kapitel 3.3
"Trainings- und Vergleichsdesign". Es wurden keine Rohdaten veraendert, keine
Trainingslaeufe gestartet, keine Modellgewichte heruntergeladen und keine
Checkpoints erzeugt.

## Zentrale Dateien fuer die Bachelorarbeit

Diese Dateien sollten erhalten bleiben und bilden den methodischen Kern des
Repositories:

- `README.md`: aktueller Repo-Ueberblick, Modellgruppen und zentrale
  Reproduzierbarkeitsregeln.
- `docs/project_context.md`: methodischer Projektkontext, globale
  Klassifikation, patchbasierte lokale Klassifikationskarten und Leakage-Risiken.
- `docs/dataset_audit_results.md`: dokumentierte Datensatzstruktur,
  Klassenverteilung, Bildformate, Gruppierungsrisiken und Konsequenzen fuer den
  Split.
- `docs/model_selection.md`: finale geplante Modellauswahl fuer Kapitel 3.2.
- `docs/training_comparison_design.md`: Trainings- und Vergleichsdesign fuer
  Kapitel 3.3.
- `docs/model_licenses.md`: methodische Lizenznotizen zu Modellgruppen,
  Gewichten und Paketquellen.
- `docs/experiment_plan.md`: grober Ablauf von Datensatzpruefung, Split,
  globalem Vergleich und patchbasierter Zusatzanalyse.
- `data/README.md`: Regeln fuer Rohdaten, Split-Manifeste, Patches und lokale
  Datensatzartefakte.

## Unterstuetzende Dateien

Diese Dateien sind nuetzlich fuer Reproduzierbarkeit und Orientierung, aber
nicht der Kerntext der Bachelorarbeit:

- `docs/dataset_audit.md`: Anleitung zum Audit-Skript und Einordnung der
  erzeugten Reports.
- `docs/dinov3_patch_smoke_test.md`: Dokumentation eines technischen
  DINOv3-Patch-Smoke-Tests ohne Training.
- `docs/dinov3_bmw25_prototype_heatmaps.md`: Dokumentation einer qualitativen
  DINOv3-Prototyp-Heatmap-Analyse.
- `configs/templates/*.yaml`: Templates fuer spaetere Experiment-Configs ohne
  echte lokale Datenpfade.
- `scripts/smoke_test_training_design.py`: prueft nur die Config-Templates mit
  Dummy-Daten.

## Prototypen und Smoke-Tests

Diese Dateien bzw. Dateigruppen sind als Prototypen, Vorstudien oder
Smoke-Tests einzuordnen:

- `scripts/audit_dataset.py`: Datensatz-Audit, kein Training.
- `scripts/dinov3_patch_smoke_test.py`: technischer DINOv3-Patch-Test.
- `scripts/dinov3_prototype_heatmaps.py`: qualitative lokale Heatmap-Analyse.
- `configs/dinov3_patch_smoke_test.yaml` und
  `configs/dinov3_bmw25_prototype_heatmaps.yaml`: vorhandene
  Prototyp-Configs, keine finalen Trainingsconfigs.
- Unversionierte Dateien wie `scripts/train_yolo_rectangle_pilot.py`,
  `scripts/prepare_yolo_rectangle_dataset.py` und die zugehoerigen
  unversionierten Configs/Dokumente sind Kandidaten fuer eine spaetere
  methodische Einordnung, Archivierung oder gezielte Versionierung.

## Sprachliche Vereinheitlichung

Die Dokumentationssprache wurde in den getrackten zentralen Markdown-Dateien
auf Deutsch vereinheitlicht. Insbesondere wurden folgende Dateien angepasst:

- `README.md`
- `docs/dataset_audit.md`
- `docs/dataset_audit_results.md`
- `docs/experiment_plan.md`
- `docs/model_licenses.md`
- `docs/model_selection.md`
- `docs/project_context.md`
- `docs/training_comparison_design.md`

Korrigiert wurden insbesondere englische Ueberschriften in
`docs/training_comparison_design.md` sowie Tabellenueberschriften in
`docs/model_selection.md`.

Zudem wurden zu starke oder veraltete Formulierungen vorsichtiger gefasst:

- Zu starke Lizenzformulierungen zum Open-Source-ViT wurden durch
  "lizenzbewusste ViT-Kontrollarchitektur" bzw. "Vision Transformer ohne
  externe vortrainierte Gewichte" ersetzt.
- DINOv3 patchbasiert wird als qualitative lokale Klassifikationskarte bzw.
  patchbasierte Heatmap-Analyse beschrieben, nicht als echte semantische
  Segmentierung.
- YOLOv11-cls wird im Hauptvergleich als Klassifikationsmodell beschrieben,
  nicht als Detektions- oder Segmentierungsmodell.
- DINOv3 Fine-Tuning bleibt optional und darf nur mit Validierungsbegruendung
  erfolgen, nicht auf Basis des Testsets.

## Klassenbezeichnungen und Tippfehler

Die vier Klassen werden in der Dokumentation konsistent als vier
Schleifgradklassen behandelt. Folgende Punkte bleiben bewusst als zu pruefen
markiert:

- In `configs/dinov3_bmw25_prototype_heatmaps.yaml` steht
  `Erste Bearbeitungssufe Viertel`. Das sieht nach einem Tippfehler fuer
  `Erste Bearbeitungsstufe Viertel` aus, kann aber ein realer lokaler
  Ordnername sein. Deshalb wurde dieser Wert nicht blind geaendert.
- `Fraeszustand` wird in Labels und Tabellen als ASCII-Schreibweise genutzt,
  waehrend lokale Ordnernamen teilweise eine Umlaut-Schreibweise enthalten
  koennen. Vor einer Aenderung muss geprueft werden, welche
  Schreibweise im lokalen Datensatz tatsaechlich existiert.
- Unversionierte Configs enthalten dieselben moeglichen Ordnernamen-Themen und
  sollten vor einer spaeteren Versionierung separat geprueft werden.

## Config-Templates

Die vier Templates unter `configs/templates/` wurden geprueft:

- `yolo11_cls_template.yaml`
- `dinov3_frozen_head_template.yaml`
- `deit_tiny_from_scratch_template.yaml`
- `dinov3_patch_analysis_template.yaml`

Sie sind als Templates gekennzeichnet, enthalten keine echten lokalen
Datenpfade, nutzen Platzhalter fuer offene Hyperparameter und enthalten die
geplante `70/15/15`-Split-Strategie, `q1`- bis `q4`-Gruppierung,
RGB-Konvertierung, Metriken und Artifact-Policy.

## Gitignore- und Repo-Hygiene

`.gitignore` wurde geprueft. Die bestehenden Regeln schliessen lokale Rohdaten,
abgeleitete Datensaetze, Patches, Feature-Caches, Gewichte, Checkpoints,
Outputs, Runs und typische grosse Archive aus. Es wurden keine Aenderungen an
`.gitignore` vorgenommen, weil die bestehenden Regeln fuer den aktuellen
Arbeitsstand ausreichen.

Bereits vorhandene unversionierte Dateien unter `configs/`, `docs/`, `reports/`
und `scripts/` wurden nicht geloescht und nicht committed. Sie sollten spaeter
gezielt geprueft werden, bevor sie versioniert, archiviert oder entfernt werden.

## Nicht loeschen

Folgende Dateien sollten ohne ausdrueckliche Entscheidung nicht geloescht
werden:

- `README.md`
- `AGENTS.md`
- `data/README.md`
- `docs/project_context.md`
- `docs/dataset_audit.md`
- `docs/dataset_audit_results.md`
- `docs/model_selection.md`
- `docs/training_comparison_design.md`
- `docs/model_licenses.md`
- `docs/experiment_plan.md`
- `configs/templates/*.yaml`
- `scripts/audit_dataset.py`
- `scripts/smoke_test_training_design.py`

## Kandidaten fuer spaetere Bereinigung

Diese Punkte bleiben offen und sollten spaeter bewusst entschieden werden:

- Unversionierte YOLO-Rechteck-Detection- und Patch-Annotationsdateien in
  `configs/`, `docs/` und `scripts/`: pruefen, ob sie fuer die Thesis relevant
  sind oder als lokale Prototypen archiviert werden.
- Unversionierte kuratierte Reports in `reports/figures/` und `reports/tables/`:
  pruefen, ob sie klein, datenschutzrechtlich unkritisch und thesisrelevant
  genug fuer eine Versionierung sind.
- Moegliche Tippfehler in lokalen Ordnernamen wie `Bearbeitungssufe` nur nach
  Abgleich mit dem echten Datensatz korrigieren.
- Falls Feature-Caches ausserhalb von `data/cache/` entstehen, sollte
  `.gitignore` gezielt erweitert werden.
