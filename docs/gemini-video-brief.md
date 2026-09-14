# Prompt pour Gemini — vidéo de démonstration

Copie tout ce qui suit le séparateur dans Gemini. Le brief est long exprès :
Gemini ne connaît pas le projet, et la qualité du script dépend entièrement de
la précision des faits qu'on lui donne. Remplace uniquement les champs `<…>`.

Un seul enregistrement d'écran sert aux deux versions (FR et EN) : le script est
écrit pour que la voix off puisse changer sans retoucher l'image.

---

You are a video director and copywriter who makes short technical demo videos
for B2B software. I need a screen-recording demo with voiceover, in two language
versions built from one single recording.

## Who I am

I am a computer-science student in Lyon (INSA Lyon + École 42) and a registered
freelance developer in France. I build custom software for small industrial
companies — sheet metal, boilermaking, precision machining, 20 to 80 employees.
I completed an internship at a 42-person sheet-metal SME near Toulouse, where I
built two production tools that are still in use. I am not selling a SaaS
product; I sell custom development and a monthly maintenance contract.

## What the video shows

A working assistant that answers questions about French labour law for an
industrial SME, and cites the exact article behind every statement.

The concrete facts, all real and measured — use these numbers, do not invent
others:

- Corpus: the French Code du travail plus the metallurgy collective agreement
  (convention collective de la métallurgie, IDCC 3248). 30 documents, 2 204
  articles, 2 634 indexed extracts. Built from French government open data
  (DILA), reproducible with one command.
- Every answer carries a clickable citation. Clicking `[1]` opens the exact
  article text underneath. The user verifies instead of trusting.
- Out-of-scope questions are refused. Asked for a tarte tatin recipe, the
  assistant says it has no information on that — and the language model is never
  even called, so it cannot invent anything.
- Measured retrieval quality on 13 labelled questions: the correct article is in
  the top 5 results 85% of the time; 4 out of 4 out-of-scope questions correctly
  refused. The refusal threshold was calibrated on data, not guessed.
- Bilingual: ask in French or English, get the answer in that language, from the
  same French corpus.
- It can run entirely on the company's own machine — no external API, no data
  leaving the building. That is a real selling point for HR documents.

## Audience and goal

Primary: owners and plant managers of French industrial SMEs. They are not
technical. They have folders of HR rules, a collective agreement nobody has
read, and a payroll question every week that takes somebody an hour to answer.
They have seen AI demos that confidently invented things, and they do not trust
them.

Secondary: technical recruiters and engineers who will read the LinkedIn post.

Goal of the video: make the viewer think "this one actually cites its sources,
and it says when it does not know" — and then click through to the live demo.

## Format constraints

- 60 to 90 seconds. Hard limit.
- Vertical 9:16 and a 16:9 version of the same cut.
- Screen recording of a real browser session, no motion graphics beyond simple
  on-screen text and highlight boxes.
- The voiceover is recorded separately, so the script must fit the shot timings
  without depending on my speaking speed.
- Silent autoplay is the default on LinkedIn: every essential point must also
  appear as on-screen text.
- Two voiceovers from one recording: French (for prospects) and English (for the
  profile and recruiters). The image, the on-screen text timings and the shot
  list are identical; only the spoken text and subtitle text change.

## What I want from you

1. **A shot list** with timestamps in seconds, saying exactly what happens on
   screen in each shot and for how long. Assume I can record anything the app
   really does; do not invent features.
2. **The French voiceover script**, timed to those shots, with a word count per
   shot so I can check the pacing.
3. **The English voiceover script**, same timings, not a literal translation —
   written natively.
4. **On-screen text** for each shot, in both languages, short enough to read in
   the time the shot lasts.
5. **Three options for the first 3 seconds** (the scroll-stopper), and your
   reasoning for which one you would pick and why.
6. **A LinkedIn post** to go with the video, in both languages, under 1 300
   characters, first line written to survive the "see more" cut.
7. **A shooting checklist**: what to have open, what to hide, browser and window
   settings, what to do about the 20-second wait while the model generates.

## Style rules

- No hype. No "revolutionary", "game-changer", "powered by AI". The audience is
  suspicious of exactly that vocabulary.
- Lead with the problem in the viewer's own words, not with the technology.
- Concrete over abstract: a real question about paid dressing time beats "boost
  your productivity".
- The refusal is the most persuasive moment in the video. Give it real screen
  time, do not bury it at the end as a footnote.
- Never claim or imply it replaces a lawyer or provides legal advice. It reads
  documents the company already has and shows where the answer comes from.
- Say honestly that it is a demonstration project built on public legal texts,
  and that the same system runs on a company's own internal documents.

## One real constraint to design around

Generation takes about 20 seconds on a local model. The sources appear after
2-3 seconds, before the text starts arriving. Do not hide this with a cut that
looks like a lie — instead use those seconds: that is when the viewer reads the
articles that were found. Tell me how you would stage it.

## Before you write

Ask me up to five questions whose answers would change the script — about the
audience, my positioning, or what I want the viewer to do next. Then write
everything.
