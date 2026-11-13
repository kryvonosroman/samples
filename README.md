# Rap Studio Sound Store

Free sound packs for **Rap Studio**. The app reads `index.json` and downloads packs from this repository.

| Pack | Style | Sounds |
| --- | --- | --- |
| Dota Rap Kit | Dota rap / drill-adjacent, 140-150 BPM | 19 |
| Drill Kit | UK / NY drill, 140-145 BPM | 10 |
| Detroit Kit | Detroit rap, 90-100 BPM | 8 |
| Boom Bap Kit | Boom bap, 85-95 BPM | 8 |
| Phonk Kit | Phonk / drift phonk, 130-160 BPM | 14 |

Every sound was synthesized from scratch for Rap Studio. Nothing is sampled from songs, games or other
sound packs. All sounds are released under **CC0 1.0**: use them in any music, free or paid, no credit needed.

## Adding a pack

1. Put the WAV files in `packs/<pack-id>/`.
2. Add the pack to `index.json` (name, style, description, and each sound's name, category and file path).
3. Push. The store in Rap Studio picks it up the next time it's opened.
