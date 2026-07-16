# DINOv3-Trainingsplan

Stand: 2026-07-16

## Ziel

Dieser Plan beschreibt den vorgesehenen DINOv3-Workflow fuer die globale
Bildklassifikation der vier Schleifgradklassen. Er dokumentiert die geplante
Validierungsstrategie, aber keine finalen Ergebnisse.

## Modellwahl

Primaerer Startpunkt ist `facebook/dinov3-vitb16-pretrain-lvd1689m`.
ViT-B/16 wird gewaehlt, weil es einen sinnvollen Kompromiss zwischen
Repraesentationsstaerke und lokalem Hardwarebedarf darstellt. ViT-L/16 wird
fuer den ersten Lauf nicht verwendet, da die lokale RTX 4060 Ti mit 8 GB VRAM
knapp werden koennte. `facebook/dinov3-vits16-pretrain-lvd1689m` bleibt als
Fallback vorgesehen, falls ViT-B/16 nicht laedt oder im Smoke-Test nicht
stabil laeuft.

## Frozen Backbone

Der DINOv3-Backbone bleibt eingefroren. Damit prueft der Ansatz bewusst, ob
die vortrainierten DINOv3-Repraesentationen fuer die bildbasierte Bewertung
des Schleifgrades metallischer Oberflaechen ausreichend trennscharf sind. Ein
vollstaendiges Fine-Tuning wuerde deutlich mehr Freiheitsgrade einfuehren und
ist fuer den ersten globalen Vergleich nicht der methodische Startpunkt.

## Linearer Head

Als Head wird zunaechst ein einzelner linearer Klassifikationskopf verwendet.
Diese Wahl haelt die Auswertung der Backbone-Repraesentation transparent und
reduziert das Risiko, dass ein komplexer Head die eigentliche Aussage ueber
DINOv3 ueberlagert. Ein MLP-Head kann spaeter optional als
Validierungsvariante geprueft werden.

## Zugriff und Artefakte

DINOv3-Gewichte duerfen nicht ins Repository. Der Standard ist
`allow_download: false`; ein Download ueber Hugging Face darf nur mit
explizitem CLI-Override erfolgen. Falls Modellbedingungen oder Zugriff fehlen,
wird der Lauf abgebrochen und der Zugriff muss ausserhalb des Repositories
geklaert werden. Hugging-Face-Tokens oder Credentials werden nicht
gespeichert.

Lokale Outputs, Feature-Caches, Predictions, Metriken, Gewichte und
Checkpoints bleiben unter ignorierten Pfaden wie `outputs/`, `data/cache/`,
`weights/` oder `checkpoints/`.

## Offene Punkte vor echtem Training

- Hugging-Face-Zugriff oder lokale Gewichtsbereitstellung fuer ViT-B/16
  klaeren.
- `--check-model` erfolgreich ausfuehren und Feature-Dimension dokumentieren.
- `--smoke-test` mit wenigen Train/Val-Bildern ausfuehren.
- Falls ViT-B/16 nicht sauber laeuft, ViT-S/16 als Fallback pruefen.
- Erst nach erfolgreichem Techniktest den echten Head-Trainingslauf starten.
- Das Testset bleibt bis zur finalen Evaluation unberuehrt.

Finale DINOv3-Ergebnisse werden spaeter separat dokumentiert.
