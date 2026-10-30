"""Synthetic German caregiver voice notes: scripted texts -> TTS audio.

Each note is written with a deliberate edge case (fillers, self-corrections,
ambiguous dosages, two residents in one note, missing information, dialect
flavour, contradictions), because the eval has to measure exactly those
failure modes. Voices and speaking rates rotate across edge-tts's German,
Austrian and Swiss neural voices to vary the speakers.

Outputs, per note:
    data/scripted/note_XX.txt  — the clean script (ablation input: clean text
                                 vs ASR output isolates speech errors)
    data/audio/note_XX.mp3     — the spoken version

Run:  uv run python data/generate_notes.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import edge_tts

DATA_DIR = Path(__file__).parent
SCRIPTED_DIR = DATA_DIR / "scripted"
AUDIO_DIR = DATA_DIR / "audio"


@dataclass(frozen=True)
class NoteScript:
    note_id: str
    text: str
    voice: str
    rate: str = "+0%"
    # What this note deliberately tests; ends up in the eval report.
    edge_case: str = ""


NOTES: list[NoteScript] = [
    NoteScript(
        "note_01",
        "So, kurze Notiz zu Frau Weber. Sie hat heute Mittag kaum etwas gegessen, "
        "vielleicht zwei, drei Löffel Suppe. Beim Aufstehen vom Stuhl ist sie fast "
        "gestürzt, ich musste sie stützen. Blutdruck war 145 zu 90. Bitte morgen "
        "nochmal nach dem Essen schauen.",
        "de-DE-KatjaNeural",
        edge_case="baseline: observation + risk + measurement + follow-up",
    ),
    NoteScript(
        "note_02",
        "Ähm, ja, also Herr Yilmaz hat heute Morgen wieder über Schmerzen im rechten "
        "Knie geklagt, so beim Laufen, ne. Ich hab ihm dann... äh... die übliche Dosis "
        "gegeben, wie besprochen. Er wollte danach nicht zur Gymnastik runter, hat die "
        "ganze Zeit auf dem Zimmer gesessen. Trinken klappt gut, die Flasche war fast leer.",
        "de-DE-ConradNeural",
        edge_case="fillers + ambiguous dosage ('die übliche Dosis')",
    ),
    NoteScript(
        "note_03",
        "Kurz zu Herrn Brandt, Zimmer zwölf. Die Stelle am Steißbein sieht heute röter "
        "aus als gestern, da müssen wir echt aufpassen. Ich hab ihn umgelagert und die "
        "Fersen freigelegt. Achso, und Frau Keller hat vorhin nach ihrer Tochter "
        "gefragt, aber das nur nebenbei. Bei Herrn Brandt bitte beim Spätdienst nochmal "
        "die Haut anschauen und dokumentieren.",
        "de-DE-AmalaNeural",
        edge_case="second resident mentioned in passing + dekubitus risk",
    ),
    NoteScript(
        "note_04",
        "Frau Krüger, Frühdienst. Alles unauffällig heute, keine Besonderheiten.",
        "de-DE-KillianNeural",
        edge_case="minimal note: almost everything should be null/empty",
    ),
    NoteScript(
        "note_05",
        "Herr Okafor hat beim Mittagessen... nein, warte, das war gestern. Also heute: "
        "Herr Okafor hat beim Frühstück seine Tabletten nicht nehmen wollen, hat sie "
        "erst liegen lassen und dann nach langem Zureden doch genommen. Sonst war er "
        "gut drauf, hat viel erzählt.",
        "de-DE-KatjaNeural",
        rate="+8%",
        edge_case="self-correction mid-sentence ('nein, warte, das war gestern')",
    ),
    NoteScript(
        "note_06",
        "Zu Frau Lindner: Sie hat heute wieder viel zu wenig getrunken, vielleicht ein "
        "halbes Glas über den ganzen Vormittag. Die Lippen waren ganz trocken. Ich hab "
        "ihr ein Trinkprotokoll hingestellt. Bitte im Spätdienst aktiv Getränke anbieten "
        "und eintragen.",
        "de-AT-IngridNeural",
        edge_case="dehydration risk + follow-up actions",
    ),
    NoteScript(
        "note_07",
        "Herr Schäfer hatte heute Nachmittag achtunddreißig Komma zwei Temperatur, "
        "gemessen um sechzehn Uhr im Ohr. Puls war zweiundneunzig. Er wirkt schlapp und "
        "hat sich gleich wieder hingelegt. Hausarzt ist informiert.",
        "de-DE-ConradNeural",
        rate="-5%",
        edge_case="spoken numbers: vitals as words, two measurements",
    ),
    NoteScript(
        "note_08",
        "Frau Petrova sagen heute Morgen, der Rücken tut sehr weh, unten, wenn sie "
        "aufstehen. Sie nicht wollen Frühstück essen deswegen. Ich geben Wärmekissen, "
        "das helfen ein bisschen. Bitte Doktor fragen wegen Schmerzmittel.",
        "de-DE-AmalaNeural",
        edge_case="non-native speaker grammar + pain risk",
    ),
    NoteScript(
        "note_09",
        "Ja also, was war denn heute... Montag war ja Friseur, das hat ihr gut gefallen, "
        "aber das war ja Montag. Heute bei Frau Albrecht eigentlich ein ruhiger Tag, "
        "wobei, eine Sache: beim Transfer vom Bett in den Rollstuhl hat sie gezittert "
        "und wollte sich nicht richtig festhalten. Wir haben dann zu zweit transferiert. "
        "Ihre Tochter kommt übrigens Samstag, hat sie dreimal erzählt, sie freut sich "
        "riesig. Ach ja, und die Heizung im Zimmer klemmt, hab ich der Haustechnik gemeldet.",
        "de-DE-KatjaNeural",
        rate="+10%",
        edge_case="long rambling note: irrelevant chatter around one real observation",
    ),
    NoteScript(
        "note_10",
        "Herr Demir hat heute Mittag richtig gut gegessen, den Teller fast leer. Beim "
        "Abendessen hat er das Essen komplett verweigert und war dabei ziemlich "
        "aufgebracht. Vielleicht morgen das Lieblingsessen probieren.",
        "de-AT-JonasNeural",
        edge_case="apparent contradiction (ate well midday, refused dinner) for the QM pass",
    ),
    NoteScript(
        "note_11",
        "Kurze Rückmeldung zu Frau Sommer: Sie hat die Augentropfen heute früh wieder "
        "abgelehnt, sagt, die brennen so. Ich hab es später nochmal versucht, da ging "
        "es dann. Bitte das Brennen beim nächsten Arztbesuch ansprechen, nicht dass da "
        "was nicht stimmt.",
        "de-CH-LeniNeural",
        edge_case="medication refusal + Swiss-accented voice",
    ),
    NoteScript(
        "note_12",
        "Wichtig: Herr Nowak ist heute gegen vierzehn Uhr dreißig im Bad gestürzt, beim "
        "Umdrehen am Waschbecken. Er ist nicht mit dem Kopf aufgeschlagen, hat aber eine "
        "Prellung am linken Ellenbogen. Blutdruck danach hundertsechzig zu fünfundneunzig. "
        "Arzt ist verständigt, Sturzprotokoll ist ausgefüllt. Bitte heute Nacht "
        "engmaschig nach ihm schauen.",
        "de-DE-ConradNeural",
        edge_case="actual fall with time, injury, measurement and protocol",
    ),
    NoteScript(
        "note_13",
        "Die Dame von Zimmer acht war heute sehr unruhig, ist immer wieder zur Tür "
        "gelaufen und wollte nach Hause. Wir haben sie mit Fotos von früher etwas "
        "beruhigen können. Nachts bitte öfter reinschauen.",
        "de-DE-KillianNeural",
        rate="-8%",
        edge_case="no name, only a room reference: resident_ref handling",
    ),
    NoteScript(
        "note_14",
        "Frau Ebert und Frau Lang hatten heute beim Mittagessen einen Streit, Frau Ebert "
        "hat danach geweint und wollte nicht mehr im Speisesaal sitzen. Sie hat dann auf "
        "dem Zimmer gegessen, aber nur die Hälfte. Frau Lang war nachher wie immer. "
        "Vielleicht die Sitzordnung ändern.",
        "de-DE-KatjaNeural",
        edge_case="two residents genuinely entangled in one incident",
    ),
    NoteScript(
        "note_15",
        "Ja grüß dich, kurz zum Herrn Huber: der hat heut wieder arg wenig gegessen, "
        "gell, bloß a bissle was vom Brei. Und beim Laufen ist er arg wackelig auf den "
        "Beinen, i hab ihn am Arm gführt. Rollator will er partout nicht nehmen, der "
        "sture Hund. Schaut's da bitte nochmal drauf.",
        "de-DE-KillianNeural",
        edge_case="southern German dialect flavour",
    ),
    NoteScript(
        "note_16",
        "Bei Herrn Janssen irgendwann nach dem Frühstück den Verband am linken "
        "Unterschenkel gewechselt. Die Wunde sieht ein bisschen besser aus als beim "
        "letzten Mal, die Ränder sind ruhiger. Er sagt, es juckt, aber es tut nicht "
        "mehr weh. Nächster Verbandswechsel übermorgen.",
        "de-CH-JanNeural",
        edge_case="vague time ('irgendwann nach dem Frühstück') + wound care",
    ),
]


async def synthesise(note: NoteScript) -> None:
    communicate = edge_tts.Communicate(note.text, note.voice, rate=note.rate)
    await communicate.save(str(AUDIO_DIR / f"{note.note_id}.mp3"))


async def generate_all() -> None:
    SCRIPTED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for note in NOTES:
        (SCRIPTED_DIR / f"{note.note_id}.txt").write_text(note.text + "\n", encoding="utf-8")
        await synthesise(note)
        print(f"{note.note_id}: {note.voice} ({note.rate}) — {note.edge_case}")


if __name__ == "__main__":
    asyncio.run(generate_all())
