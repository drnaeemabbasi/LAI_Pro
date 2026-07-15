# laipro — User Guide

A complete, step-by-step guide to installing and using **laipro** to measure leaf area (PAI / LAI / FCOVER) from downward-looking fish-eye crop photographs. No prior experience needed — just follow the steps in order.

---

## Contents

1. [What laipro does](#1-what-laipro-does)
2. [What you need before you start](#2-what-you-need-before-you-start)
3. [Part A — Install laipro (one time)](#3-part-a--install-laipro-one-time)
4. [Part B — Get your photos ready](#4-part-b--get-your-photos-ready)
5. [Part C — Run laipro (the easy way, with the app)](#5-part-c--run-laipro-the-easy-way-with-the-app)
6. [Part D — Run laipro (command line)](#6-part-d--run-laipro-command-line)
7. [Part E — Understand your results](#7-part-e--understand-your-results)
8. [Part F — Optional: improve the results](#8-part-f--optional-improve-the-results)
9. [Tips for accurate, comparable results](#9-tips-for-accurate-comparable-results)
10. [Troubleshooting](#10-troubleshooting)
11. [Glossary of terms](#11-glossary-of-terms)
12. [Getting help](#12-getting-help)

---

## 1. What laipro does

You take straight-down photos of a crop plot with a **fish-eye lens** (camera held level, pointing at the ground). laipro reads a folder of those photos and calculates how much leaf/canopy is there:

- **FCOVER** — how much of the ground is covered by vegetation (%)
- **PAI** — plant area index (amount of canopy; "effective" and clumping-corrected "true")
- **ALA** — average leaf angle
- **Clumping index**, **LAI57**, and **FAPAR** (light absorbed by the canopy)

It produces a tidy **HTML report** plus spreadsheet files, and records exactly how each result was made so it's fully reproducible.

*(Definitions of every term are in the [Glossary](#11-glossary-of-terms).)*

---

## 2. What you need before you start

- A **Windows PC**.
- Your **fish-eye photos** copied onto that PC.
- About **15 minutes** for the one-time install.
- An internet connection (for the install only).

That's it. You do **not** need a graphics card.

---

## 3. Part A — Install laipro (one time)

You only do this once per computer.

### Step 1 — Install Python

1. Go to **https://www.python.org/downloads/**.
2. Download **Python 3.12** (recommended). *(3.11 also works. Avoid the very newest version.)*
3. Run the installer. On the first screen, **tick the box "Add Python to PATH"**, then click **Install Now**.

> ⚠️ The "Add Python to PATH" checkbox is easy to miss and important. If you forget it, reinstall and tick it.

### Step 2 — Download the laipro code

1. Go to the laipro GitHub page.
2. Click the green **Code** button → **Download ZIP**.
3. Unzip it somewhere easy to find, e.g. `C:\laipro`.

*(Or, if you use git: `git clone <repo-url>`.)*

### Step 3 — Open a terminal in the laipro folder

1. Open the unzipped `laipro` folder in File Explorer.
2. Click the address bar, type `powershell`, and press **Enter**. A blue/black terminal window opens, already in the folder.

### Step 4 — Create the environment and install

Copy-paste these three lines into the terminal, pressing **Enter** after each:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install ".[gui]"
```

- The first line makes a private workspace. The second turns it on (your prompt now starts with `(venv)`). The third downloads and installs laipro and the app — this takes a few minutes.

> ⚠️ If the second line gives a red error about "running scripts is disabled," run this **once**, then repeat the second line:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> (Answer `Y` if it asks.)

### Step 5 — Check it worked

```powershell
laipro --version
```

If you see something like `laipro 0.1.0`, you're done. 🎉

> From now on, every time you use laipro you open PowerShell in the laipro folder and run `.\venv\Scripts\Activate.ps1` first (see [Step 3–4](#step-3--open-a-terminal-in-the-laipro-folder)). That's the only repeat step.

---

## 4. Part B — Get your photos ready

### How the photos should be taken

For valid results, each photo should be:

- Taken with the **fish-eye lens**, camera held **level** (use the bubble level) and pointing **straight down** at the canopy.
- **In focus** and **not over-exposed** (no blown-out white patches).
- Taken about **80 cm above the canopy**, moving between/across rows.

### How to organize them

- Put **all photos of one plot in one folder**. That folder is one "site" / plot.
- Use a clear folder name, e.g. `Field3_20260714_Wheat`.
- Keep different plots in different folders — you run laipro once per folder.

Example:
```
C:\MyLAI\
   Field3_Wheat\      <- one plot: put its photos here
      DSC_3155.NEF
      DSC_3155.JPG
      DSC_3157.NEF
      ...
   Field7_Corn\       <- another plot
      ...
```

> laipro reads RAW files (`.NEF`, `.CR2`, …) and regular photos (`.JPG`, `.TIFF`, `.PNG`). If both a RAW and a JPG exist for the same photo, it uses the RAW automatically.

---

## 5. Part C — Run laipro (the easy way, with the app)

This is the simplest path — a window with buttons.

### Step 1 — Launch the app

In your terminal (with `(venv)` showing):

```powershell
laipro gui
```

A window titled **laipro** opens. On the **right** is a panel of buttons. In the **center** is where your photo will show. *(First launch takes a few seconds.)*

### Step 2 — Open your plot folder

1. In the right panel, next to **plot folder**, click the **…** button.
2. Select your plot folder (e.g. `Field3_Wheat`) and confirm.
3. Click **Open**.

Your first photo appears, and the title bar shows something like `1 / 12`.

### Step 3 — (Optional but recommended) Prepare

This decodes your photos once so processing is faster next time.

1. Leave **working resolution** at **2500**.
2. Click **Prepare (decode + cache)**.
3. A progress bar runs along the bottom. Wait for it to finish.

*(You only need to do this once per folder.)*

### Step 4 — Process

1. Click **Process folder (DHP)**.
2. A progress bar shows each photo being processed. When it finishes, the bottom status bar shows your results (PAI, FCOVER, …).

> The window stays responsive while it works — that's normal. Just wait for the progress bar.

### Step 5 — Look at your results

Open the report in your browser:

1. In File Explorer, go into your plot folder → `laipro_results`.
2. Double-click **`report.html`**.

You'll see the headline numbers, a per-photo table, plots, and the detection overlays. *(What each number means is in [Part E](#7-part-e--understand-your-results).)*

**That's the whole basic workflow: Open → Process → open the report.** Everything below is optional.

---

## 6. Part D — Run laipro (command line)

If you prefer typing, or want to process many folders quickly, you don't need the app. With `(venv)` active:

```powershell
laipro process C:\MyLAI\Field3_Wheat --latitude 50.1 --doy 195
```

- Replace the path with your plot folder.
- `--latitude` = your site's latitude; `--doy` = day of the year the photos were taken (1–365). **These are only needed for the FAPAR number.**

Speed up repeated runs by preparing once first:

```powershell
laipro prepare  C:\MyLAI\Field3_Wheat
laipro process  C:\MyLAI\Field3_Wheat --latitude 50.1 --doy 195
```

See all options any time with:

```powershell
laipro --help
laipro process --help
```

Results go to the same `laipro_results` folder as the app.

---

## 7. Part E — Understand your results

After processing, your plot folder contains a **`laipro_results`** folder with:

| File | What it is |
|---|---|
| **report.html** | Open this first — the visual summary (numbers, plots, overlays) |
| **results.json** | The plot's headline numbers, for records/scripts |
| **per_image.csv** | One row per photo (vegetation %, FCOVER, LAI57) — opens in Excel |
| **gap_fraction_by_ring.csv** | Advanced: gap fraction by view angle |
| **qc/** | One diagnostic picture per photo (see below) |
| **provenance.json** | Exact settings + file fingerprints (reproducibility) |

### The headline numbers (in report.html)

| Number | Plain meaning |
|---|---|
| **FCOVER** | % of ground covered by canopy, looking straight down |
| **PAI (effective / true)** | How much canopy there is; "true" corrects for clumping |
| **ALA** | Average leaf angle (0° = flat, 90° = upright) |
| **Clumping** | How clumped the foliage is (1 = random, <1 = clumped) |
| **FAPAR** | Fraction of sunlight the canopy absorbs |

### The QC pictures (`qc/` folder)

Each photo gets a 4-panel diagnostic image so you can *see* what was detected:

- **top-left:** your original photo
- **top-right:** detected vegetation shown in **teal** (with the vegetation % printed)
- **bottom-left:** a black-and-white map (white = vegetation, black = soil, grey = ignored edges)
- **bottom-right:** a histogram showing where the plant/soil cut-off was set

Use these to confirm the teal is landing on the actual plants. Some soil showing between rows is normal and correct.

---

## 8. Part F — Optional: improve the results

The default detection is usually very good, so **most people can skip this section.** Use it only if the QC pictures show the teal missing plants or grabbing soil.

### Improve vegetation detection (teach it)

In the app:

1. Click the **veg labels** layer (left-hand layer list).
2. Click the **paintbrush** tool (top-left).
3. Set the label number to **2** and paint over any plants it missed; set it to **1** and paint over any soil it wrongly marked. A few strokes is enough.
4. Set **segmenter dir** to a name like `models\wheat`, then click **Train segmenter**.
5. Move to the next photo (**Next**), correct a bit more, **Train segmenter** again — it keeps learning.

> Paint and click **Train** *before* moving to the next photo — changing photos clears your paint.

### Hide the operator's boots / monopod

If your feet or the pole appear in shots:

1. Click the **object polygons** layer.
2. Click the **rectangle** tool and **drag a box** around the boot. *(Rectangle is easiest. To use the polygon tool instead: click each corner, then **double-click** to finish — do not press Escape, that cancels it.)*
3. Set **mask dir** to e.g. `models\rig`, then click **Train mask**.
4. Click **Auto-mask (predict object)** on other photos to check it — it excludes only the object, not a whole strip.

### Use your trained models later (command line)

```powershell
laipro process C:\MyLAI\Field7 --model models\wheat --mask-model models\rig --latitude 50 --doy 200
```

---

## 9. Tips for accurate, comparable results

- **Use the same settings for every plot** you want to compare (resolution, latitude/DoY handling). Consistency matters more than any single setting.
- **Resolution:** the default (2500) is the sweet spot — full resolution is not more accurate and is slower. Don't go below ~1500.
- **FAPAR** needs the correct **`--latitude` and `--doy`** (in the app these use defaults, so for exact FAPAR use the command line). FCOVER, PAI, ALA, and clumping do **not** depend on these.
- **Results are PAI, not pure LAI** (photos can't separate leaves from stems) — this is normal and matches satellite products.
- Treat the numbers as high-quality **optical estimates**, excellent for comparing plots, dates, and treatments.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Python wasn't added to PATH. Reinstall Python and tick **"Add Python to PATH."** |
| "running scripts is disabled" when activating | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (answer `Y`), then retry. |
| `laipro` not recognized | Make sure `(venv)` shows in your prompt (`.\venv\Scripts\Activate.ps1`). If still missing, run `pip install ".[gui]"` again. |
| The app window won't open | Reinstall the app part: `pip install ".[gui]"`. Then `laipro gui`. |
| "No images found" | You pointed at the wrong folder, or the photos are in a sub-folder. Put the photos directly in the folder you select. |
| Processing seems stuck | Watch the progress bar at the bottom — large RAW photos take a few seconds each. It is working. |
| FAPAR looks wrong | Set the real `--latitude` and `--doy` on the command line. |

---

## 11. Glossary of terms

| Term | Meaning |
|---|---|
| **DHP** | Digital Hemispherical Photography — using a fish-eye photo to measure canopy from the pattern of gaps. |
| **FCOVER** | Fraction of ground covered by vegetation, looking straight down (0–100%). |
| **LAI** | Leaf Area Index — leaf area per unit ground area. |
| **PAI** | Plant Area Index — like LAI but counts all plant parts (what a photo actually measures). |
| **Effective vs True PAI** | Effective assumes leaves are scattered randomly; True corrects for clumping. |
| **GAI** | Green Area Index — PAI of only the green parts. |
| **ALA** | Average Leaf Angle (0° flat, 90° upright, ~57° random). |
| **Clumping index** | How non-random the foliage is (1 = random, <1 = clumped). |
| **Gap fraction** | Fraction of gaps you can see through the canopy in a direction. |
| **LAI57** | A leaf-area estimate from the 57.5° view angle that needs no leaf-angle assumption. |
| **FAPAR** | Fraction of Absorbed Photosynthetically Active Radiation (sunlight the canopy absorbs). |
| **Nadir** | Straight down (0°). |
| **Zenith angle** | Angle away from straight-down; the fish-eye captures many angles at once. |
| **COI** | Circle of Interest — the central part of the image that is analysed (edges are ignored). |
| **RAW / NEF** | The camera's uncompressed file (NEF = Nikon). Best quality. |

---

## 12. Getting help

- Run `laipro --help` or `laipro <command> --help` for command options.
- Open a photo's picture in the `qc/` folder to see exactly what was detected.
- If something fails, note the exact error message from the terminal — it usually points straight to the fix in [Troubleshooting](#10-troubleshooting).
