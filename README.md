# Rap Studio Sound Store

Free sound packs for **Rap Studio**. The app reads `index.json` and downloads packs from this repository.

| Pack | Style | Sounds |
| --- | --- | --- |
| Dota Rap Kit | Dota rap / drill-adjacent, 140-150 BPM | 19 |
| Drill Kit | UK / NY drill, 140-145 BPM | 10 |
| Detroit Kit | Detroit rap, 90-100 BPM | 8 |
| Boom Bap Kit | Boom bap, 85-95 BPM | 8 |
| Phonk Kit | Phonk / drift phonk, 130-160 BPM | 14 |
| Zxc Kit | Zxc / emo-trap, 140 BPM (guitar and pad loops, vocal chops) | 15 |
| Memphis Phonk Kit | Memphis phonk, 120-135 BPM | 14 |

Every sound was synthesized from scratch for Rap Studio. Nothing is sampled from songs, games or other
sound packs. All sounds are released under **CC0 1.0**: use them in any music, free or paid, no credit needed.

## Kits by other makers (links only)

The store also lists phonk kits by other makers under `links` in `index.json`. Those sounds are **not**
stored here: their licences allow using them in your music, not re-sharing the files. In Rap Studio you
press **Get it** to open the maker's page, download the zip there, then press **Add zip...** to copy its
sounds into your Browser. Check each kit's licence before releasing a song.

| Kit | Maker | Price |
| --- | --- | --- |
| Lunatic Phonk | BVKER | Free |
| VIVIVI Phonk Drum Kit | VIVIVI | Free |
| Looperman Phonk Loops | Looperman community | Free (account) |
| Slooply Free Phonk Samples | Slooply | Free |
| Phonk Drum Shots | GVLXY on SampleFocus | Free (account) |
| Reticence Phonk Drum Kit | BVKER | about $5 |
| DRIFT Phonk Drum Kit | VILARCORP | about EUR 5 |
| Rage Phonk Drum Kit | drumkit | EUR 5+ |

## Adding a pack

1. Put the WAV files in `packs/<pack-id>/`.
2. Add the pack to `index.json` (name, style, description, and each sound's name, category and file path).
3. Push. The store in Rap Studio picks it up the next time it's opened.
