# Untersuchung von Vision-Transformer-Modellen zur bildbasierten Bewertung des Schleifgrades metallischer Oberflaechen

Dieses Repository begleitet die experimentelle Durchfuehrung einer Bachelorarbeit zur bildbasierten Klassifikation des Schleifgrades metallischer Oberflaechen. Ziel ist ein reproduzierbarer Vergleich moderner Bildklassifikationsverfahren, insbesondere Vision-Transformer-basierter Modelle, unter besonderer Beachtung sauberer Datensplits und der Vermeidung von Data Leakage.

## Aktueller Status

Dieses Repository enthaelt zunaechst nur Grundstruktur, Dokumentation und eine initiale Abhaengigkeitsliste. Trainingscode wird erst ergaenzt, nachdem die konkrete Datensatzstruktur geprueft und die Split-Strategie festgelegt wurde.

## Ziel der Arbeit

Untersucht werden Modelle, die aus Bildern metallischer Oberflaechen den Schleifgrad vorhersagen. Die Experimente sollen zeigen, welche Modellgruppe fuer die globale Bildklassifikation geeignet ist und ob patchbasierte lokale Vorhersagen eine sinnvolle Annahme fuer raeumlich aufgeloeste Schleifgradkarten liefern koennen.

## Geplante Modellgruppen

1. **YOLOv11-cls**  
   Praxisnahe moderne Baseline fuer globale Schleifgradklassifikation auf Bildebene.

2. **DINOv3 + Klassifikationskopf**  
   DINOv3 wird zuerst als eingefrorener Feature Extractor genutzt. Darauf wird ein einfacher Klassifikationskopf trainiert, um zu pruefen, ob die self-supervised vortrainierten Repraesentationen fuer den Schleifgrad geeignet sind.

3. **Lizenzsicherer Open-Source Vision Transformer**  
   Vergleich mit einem unkritisch nutzbaren Vision Transformer. Die konkrete Modell- und Gewichtsquelle muss vor der Nutzung in [docs/model_licenses.md](docs/model_licenses.md) dokumentiert werden.

4. **DINOv3 patchbasiert**  
   Lokale Schleifgradvorhersage als methodisch saubere Annaeherung an Segmentierung. Da keine pixelgenauen Segmentierungsannotationen vorliegen, wird keine echte semantische Segmentierung trainiert. Stattdessen werden Bilder in Patches zerlegt, patchweise klassifiziert und anschliessend als raeumliche Schleifgradkarte bzw. Heatmap visualisiert.

## Zentrale Reproduzierbarkeitsregeln

- Alle Experimente sollen ueber versionierte Config-Dateien in `configs/` steuerbar sein.
- Train/Val/Test-Splits muessen vor jeder Patch-Erzeugung auf Originalbild-, Bauteil- oder Aufnahmeebene erfolgen.
- Patches eines Originalbildes, Bauteils oder derselben Aufnahme duerfen nicht ueber verschiedene Splits verteilt werden.
- Das Testset darf nicht fuer Modellauswahl, Hyperparameter-Tuning, Schwellenwertwahl oder fruehe Designentscheidungen verwendet werden.
- Seeds, Config, Git-Commit, Paketversionen und relevante Hardwareinformationen muessen pro Experiment dokumentiert werden.
- Rohdaten, Checkpoints, Modellgewichte, Caches und grosse Outputs duerfen nicht in GitHub eingecheckt werden.

## Repository-Struktur

```text
configs/              # Versionierte Experimentkonfigurationen
data/                 # Datensatzdokumentation und ggf. Split-Manifeste, keine Rohdaten in Git
docs/                 # Methodische Dokumentation und Lizenznotizen
notebooks/            # Explorative Analysen, keine produktionskritische Logik
scripts/              # Kommandozeilen-Einstiegspunkte, spaeter config-basiert
src/
  data/               # Datenpruefung, Split- und Dataset-Logik
  models/             # Modellaufbau
  training/           # Trainingsschleifen
  evaluation/         # Metriken, Auswertung, Visualisierung
  utils/              # Gemeinsame Hilfsfunktionen
reports/
  figures/            # Kuratierte Abbildungen fuer Bericht/Thesis
  tables/             # Kuratierte Tabellen fuer Bericht/Thesis
outputs/              # Lokale Experimentoutputs, nicht fuer GitHub
```

## Naechste Schritte

1. Datensatzstruktur erfassen: Klassen, Anzahl Bilder, Aufnahmebedingungen, Bauteil-IDs, Wiederholungsmessungen.
2. Leakage-Risiken identifizieren: gleiche Bauteile, gleiche Aufnahmeserien, stark ueberlappende Bildausschnitte, Patches aus demselben Originalbild.
3. Split-Strategie festlegen und dokumentieren.
4. Erst danach Dataset- und Trainingscode implementieren.

Fuer die fruehe strukturelle Datensatzpruefung in Kapitel 3.1 steht
[docs/dataset_audit.md](docs/dataset_audit.md) zur Verfuegung.

Weitere Details stehen in [docs/project_context.md](docs/project_context.md), [docs/experiment_plan.md](docs/experiment_plan.md) und [data/README.md](data/README.md).
