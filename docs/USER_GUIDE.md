# User Guide

This guide is for people who want to use the app, not program it. It takes about ten
minutes to read, and you need no background in statistics or computing.

> This is a research and education tool, provided "AS IS" without warranty. Results are
> statistical estimates, not professional advice. See [DISCLAIMER.md](../DISCLAIMER.md).

Contents:

1. [What the app does](#1-what-the-app-does)
2. [Your first run](#2-your-first-run)
3. [Finding your way around](#3-finding-your-way-around)
4. [Building your own test](#4-building-your-own-test)
5. [Which settings work well together](#5-which-settings-work-well-together)
6. [How to read the results](#6-how-to-read-the-results)
7. [Using real models](#7-using-real-models)
8. [Tips for good tests](#8-tips-for-good-tests)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary](#10-glossary)

## 1. What the app does

Ask a person the same question on Monday and again on Friday, and you may get two different
answers. Daniel Kahneman and his co-authors call this unwanted variation *noise*. AI models
show it too.

The app asks one question many times. Each time it opens a new conversation, so the model
cannot see what it said before. We call one such conversation a repetition. Afterwards the
app measures three things:

- Noise: how much the answers scatter.
- Bias: whether the answers are always off in the same direction. For this you have to know
  the correct answer.
- Effects: whether a detail that should not matter, such as a random number, the wording or
  the order of the facts, changes the answers.

## 2. Your first run

You need no account and no key for this. The app comes with three practice models. They are
simulations, they cost nothing and they answer at once.

1. Open the app. In Codespaces it starts by itself. Otherwise run `streamlit run app.py`.
2. You are on the page **Start here**. The first test, "Does a model give the same answer
   every time?", is already selected.
3. Press **Run the test** under step 2.
4. Read the summary that appears under step 3.

For charts and all the numbers, click the tab **3 · Results** at the top of the page.

You should see that *Steady Sage* scatters very little, *Noisy Novice* scatters a lot and
*Anchored Optimist* answers higher than the other two. They were built that way. It is a
safe way to learn what results look like before you test a real model.

To try another test, click **Choose this test** on one of the other cards and press **Run
the test** again.

## 3. Finding your way around

The sidebar on the left holds the settings. The defaults work, so you can ignore it at
first. It has a switch for practice mode, the list of models to test, a slider for the
number of repetitions per model and a slider for the randomness of the answers (called
temperature). Further down you can enter API keys for real models. Less common settings sit
under **More settings**. At the bottom, **App: restart or stop** does what it says.

The tabs at the top are the pages of the app:

- **Start here** has the ready-made tests, the run button and a short summary of the last
  result. For many purposes you never have to leave it.
- **1 · Question** is where you write your own question, choose the form of the answer and
  decide what changes between the repetitions. A preview at the bottom shows exactly what
  each model will receive.
- **2 · Analysis & run** is where you choose the analysis. The app checks whether your
  settings fit together, and you can start the run here as well.
- **3 · Results** shows the result in one or two sentences, then the numbers, charts, every
  single answer and the downloads.
- **More models** is for installing free local models or connecting a model of your own.
  You do not need it to try the app.

Most settings have a small question mark next to them. Move the mouse over it for a short
explanation.

The app follows the light or dark mode of your system. To switch by hand, open the menu at
the top right and choose **Settings → Theme**.

## 4. Building your own test

Go to **1 · Question** and work through its three parts.

First, your question. Ask for one judgement, the way you would ask a person: "Estimate the
probability that …" works better than an open question. You do not have to explain the
answer format, because the app adds that for you. If you want to test several questions at
once, choose *Several questions (CSV file)* and upload a table with the columns `case_id`
and `prompt`. A column `ground_truth` with the correct answers is optional. An example file
can be downloaded on the same page.

Second, the answer format. Answers can only be compared if they all have the same form. You
can choose between yes or no, a percentage, any number, a rating on a scale, one of several
categories, a ranking, and a decision plus confidence. If you know the correct answer, tick
the box and enter it.

Third, what changes between the repetitions:

- *Nothing.* Every repetition is identical. This is the plain consistency check.
- *The order of the facts.* You enter a few facts and the app shuffles them.
- *An irrelevant number is added (anchoring).* Half of the repetitions get an extra sentence
  with a number that has nothing to do with the question.
- *The wording (framing).* Half of the repetitions state a fact one way, the other half
  state the same fact another way.
- *The role the model plays (persona).* The model is told to be, say, a cautious analyst in
  one repetition and an optimist in the next.

Then go to **2 · Analysis & run**, choose what you want to find out and read the settings
check. A green message means everything fits. A red one tells you what to change, and the
run button stays grey until you have. Then press **Run the test**.

## 5. Which settings work well together

### 5.1 Start from your question

These seven combinations are the ready-made tests on the start page.

| Your question | Answer format | Change between repetitions | Analysis | Repetitions |
|---|---|---|---|---|
| Does a model give the same answer every time? | Percentage | Nothing | Mean pairwise distance & RMS noise | 10 |
| How far off is a model, and is that bias or noise? | Percentage or any number, with the correct answer | Nothing | MSE noise decomposition | 10 |
| Why do different models disagree? | Percentage, any number or rating | Nothing, or the role | Noise audit (several questions, at least 2 models) | 10 |
| Does an irrelevant number sway the answer? | Percentage or any number | Irrelevant number | Anchoring & framing effect size | 20 or more |
| Does the wording change the answer? | Anything except ranking | Wording | Anchoring & framing effect size | 20 or more |
| Does a model flip its yes/no decision? | Yes or no, or a category | Nothing | Categorical stability | 20 |
| Does a model reshuffle its priorities? | Ranking | Order of the facts | Ranking stability | 12 |

### 5.2 Answer format and analysis

"yes" means the pair fits, "roughly" means the result is only an approximation, and an empty
cell means it does not work.

| Answer format | MSE decomposition | Pairwise distance | Noise audit | Categorical stability | Anchoring & framing | Ranking stability |
|---|---|---|---|---|---|---|
| Yes or no | yes | yes | roughly | yes | yes | |
| Percentage | yes | yes | yes | | yes | |
| Any number | yes | yes | yes | | yes | |
| Rating on a scale | yes | yes | yes | roughly | yes | |
| One of several categories | | | | yes | yes | |
| Ranking | | | | | | yes |
| Decision plus confidence | yes | yes | yes | | yes | |

### 5.3 The change between repetitions and the analysis

| What changes | Suitable analyses |
|---|---|
| Nothing | Pairwise distance, MSE decomposition, Noise audit, Categorical stability, Ranking stability |
| The order of the facts | Ranking stability, also Pairwise distance and Categorical stability |
| An irrelevant number | Anchoring & framing effect size only |
| The wording | Anchoring & framing effect size only |
| The role the model plays | Noise audit, Pairwise distance, Categorical stability |

The anchoring and wording tests create two groups of answers on purpose. A noise analysis
would count that intended difference as noise and mislead you, which is why only the effect
size analysis fits them.

### 5.4 Values that work well

Use 10 repetitions per model for a noise check and 20 or more for the anchoring and wording
tests, because those split the repetitions into two halves. With very few answers, the
statistics are shaky.

Leave the randomness (temperature) at 0.7. At 0 most models simply repeat themselves, so
they look more consistent than they are. Above 1.2 the answers become erratic and often
break the answer format.

Two or three models are enough for a comparison. The noise audit needs at least two.

One question is fine for a quick check. The noise audit needs three or more, because the
different causes of disagreement can only be told apart across several questions.

The number of requests is models × questions × repetitions. Three models, three questions
and ten repetitions make 90 requests. The app shows the number, an estimated cost and the
expected duration before you start.

## 6. How to read the results

Read the tab **3 · Results** from top to bottom. It starts with the essentials and gets more
detailed further down.

1. Answer quality. A green, yellow or red box says what share of the answers had the
   requested form. Below about 90 % the results may be distorted, so look at the single
   answers.
2. In short. The main result in one or two sentences.
3. The numbers. Move the mouse over the question mark next to a number to see what it means.
4. Per model. One row per model. Move the mouse over a column title for its meaning.
5. Charts. A line above them says how to read them.
6. Every single answer. Useful when you want to see what a model actually wrote.

### 6.1 The analyses one by one

**Mean pairwise distance & RMS noise** answers how much the answers differ from each other.
The mean pairwise distance is the typical gap between two answers of the same model. Close
to 0 is good. *RMS noise % of scale* puts the scatter in relation to the answer scale: below
5 % is small, above 15 % is severe. If the distance between models is much larger than the
distance within a model, the choice of model matters a lot.

**MSE noise decomposition** answers how far off the answers are and why. It needs the
correct answer. A positive mean bias means the model answers too high, a negative one too
low. The RMSE is the typical total error in the unit of the answer, and lower is better. The
noise share tells you what kind of error it is: near 1 the model is right on average but
unreliable, near 0 it is consistent but wrong.

**Noise audit** answers why judges disagree. It needs at least two questions and two
models. The result names the largest of three components:

| Largest component | What it means | What helps |
|---|---|---|
| Level noise | One model is always stricter or always higher | Calibrate it, for instance by subtracting an offset |
| Stable pattern noise | Models react differently to particular questions | Hard to fix. Compare the questions |
| Occasion noise | The same model gives different answers to the same question | Lower the temperature or average several answers |

**Categorical stability** answers whether the model changes its mind. Mode agreement is the
share of repetitions that gave the most common answer, so close to 100 % is good. The flip
rate is the opposite. With two options, 50 % means the model might as well toss a coin. A
normalised entropy near 0 means always the same answer, near 1 means random.

**Anchoring & framing effect size** answers whether an irrelevant change shifted the
answers. The shift is the average of the second group minus the average of the first, in the
unit of the answer. Cohen's d expresses how large that is: 0.2 is small, 0.5 medium, 0.8 or
more large. A corrected p-value below 0.05 means the difference is unlikely to be chance. A
large effect with a p-value above 0.05 usually means you had too few repetitions.

**Ranking stability** answers whether the model keeps the same order. The Kendall tau
distance is the share of item pairs that two repetitions order differently: close to 0 is
good, 0.5 is as different as random. Kendall's W measures overall agreement, and above 0.7
is strong.

### 6.2 Three things to keep in mind

Few repetitions mean a lot of uncertainty. With five answers, a difference can easily be
chance. Columns ending in `_ci95` give a confidence interval, and a wide one means the value
is uncertain.

The numbers describe this one setup. A different question, temperature or model version can
give different results.

The practice models are simulations. Use them to learn, and use real models when you want to
find something out.

## 7. Using real models

Cloud models from OpenAI, Anthropic, Gemini and OpenRouter need an API key. That is a kind
of password you get from the provider, and the provider bills you for each request. Open
**Use real AI models (API keys)** in the sidebar, enter the key and switch practice mode
off. In Codespaces you can store keys as Codespaces Secrets instead.

Local models are free. Open the tab **More models**, install Ollama and download a small
model such as `gemma3:270m`. They run on the machine itself and are slow without a graphics
card.

If you run a model server of your own, connect it in the tab **More models** as well. See
[BRING_YOUR_OWN_MODEL.md](BRING_YOUR_OWN_MODEL.md).

Before you spend money, set a spending limit with your provider and start with five
repetitions. Do not send personal or confidential data.

## 8. Tips for good tests

Change one thing at a time. Otherwise you cannot tell what caused a difference.

Look at the preview in **1 · Question** before a paid run. It shows exactly what each model
receives.

If you want to repeat a test later, download the settings of the run from the results tab
and leave the random seed under **More settings** as it is.

After a run you can try another analysis without paying again. Switch the analysis in
**2 · Analysis & run** and press **Recalculate without asking again**. No new requests are
sent.

## 9. Troubleshooting

**The run button is grey.** A yellow or red message near the button says why. Usually no
model is selected, a field is empty, or the settings check found a combination that cannot
work. If you edited the analysis code, it may have failed the security check.

**"This analysis needs at least 2 different questions".** In **1 · Question**, choose *Several questions
(CSV file)*. The built-in example file has three.

**"This analysis needs the correct answer".** Tick *I know the correct answer* in
**1 · Question**, or upload a file that has a `ground_truth` column.

**"This analysis compares two groups of answers".** In **1 · Question**, choose the
irrelevant number or the wording as the change between repetitions.

**A model writes 0.42 instead of 42 %.** The app handles this. Fractions are rejected and
the model is reminded once in the same conversation. The column `format_retries` counts
these reminders, and both replies stay visible in `raw_output`.

**Answer quality is low.** In the results tab, tick *Show only the answers with problems*
and read the column `raw_output`. Lower the temperature, simplify the question or try
another model.

**"… model calls failed".** The provider or server refused the request. The results tab
shows its message and a hint. HTTP 401 or 403 means the key was rejected. 404 means the
model name is unknown or the model was retired; add a current name under *More settings →
Add another cloud model*. 400 means the model does not accept one of the settings. 429 means
you hit a rate limit or ran out of credit.

**"… does not accept `temperature`".** Some models refuse a setting such as temperature,
seed or JSON mode. The app repeats the request without it and lists it in the column
`dropped_params`. The model then uses its own default, so be careful when you compare its
noise with other models.

**Answers come back empty.** A model that "thinks" before answering used up its whole
length limit on thinking. Raise *Longest allowed answer (tokens)* under **More settings**.

**Noise is exactly 0.** The temperature is probably 0, or the provider returns stored
answers. Set it to 0.7.

**Only practice models are listed.** No API key or local model was found. Add a key in the
sidebar or install a model in **More models**.

**"Analysis process exit code 1 … No module named …".** The Python environment is
incomplete. Rebuild the Codespace or Dev Container, or reinstall `requirements.txt`.

**The app does not open in Codespaces.** Open the **PORTS** panel and click the globe icon
next to port 8501. Keep the port private. If nothing answers there, the app is not running:
press Ctrl+Shift+B in VS Code to start it.

**The app cannot be opened from another computer.** That is intended. The app has no login
and therefore only answers on the machine it runs on (`127.0.0.1`). In Codespaces and Dev
Containers, VS Code forwards the port to your browser.

**"api_key_env … holds a key for another service".** A server of your own needs a key
variable of its own, for example `MY_SERVER_KEY`. The app does not send your OpenAI,
Anthropic, Gemini, OpenRouter or GitHub key to other servers.

**"an API key is only sent to a public endpoint over https://".** Use the `https://` address
of the server, or remove the key variable if the server needs none.

**"'.agg' must be called directly" in the security check.** In your own analysis code,
write `df.agg(...)`, `df.apply(...)` and similar in one piece. Storing such a method in a
variable first (`f = df.agg`) is not allowed.

**"sandbox: … is not allowed".** Your analysis code tried to open a file, use the network or
start a program. The analysis only works with the answers table it receives.

**The app hangs, shows an `ImportError` or behaves oddly after an update.** Restart it. In
the app, open *App: restart or stop* at the bottom of the sidebar and press **Restart**. If
the page itself no longer works, press Ctrl+Shift+B in VS Code, or type
`bash scripts/start_app.sh restart` in a terminal. Wait about ten seconds and reload the
page. Results you have not downloaded are lost by a restart.

**I want to end the app.** Press **Stop** in the same place, or run
`bash scripts/start_app.sh stop`. In Codespaces, stopping the codespace itself (on
github.com/codespaces) ends everything and stops it using up your free hours.

## 10. Glossary

| Term | Meaning |
|---|---|
| Repetition | One new, separate conversation with a model. The code and the exported files call it a context window. |
| Question (case) | One thing you ask. In a file, one row. |
| Judge | Whoever gives the answers: a model, or a model playing a role. |
| Noise | Unwanted variation. The same question gets different answers. |
| Bias | A consistent error. Answers are always too high or always too low. |
| Ground truth | The correct answer, if it is known. |
| Occasion noise | The same judge answers the identical question differently at different times. |
| Level noise | Judges differ in their general level. One is always stricter. |
| Pattern noise | Judges react differently to particular questions. |
| Anchoring | Being pulled towards a number that has nothing to do with the question. |
| Framing | Answering differently when the same fact is worded differently. |
| Temperature | How much randomness the model uses when it writes. At 0 it is nearly deterministic. |
| Token | The unit in which models count text, roughly three quarters of a word. |
| Parse success | The answer had the requested form, so the app could read it. |
| p-value | The chance of seeing a difference this large if there were no real effect. |
| Effect size | How large a difference is, regardless of how many answers were collected. |
| Confidence interval (CI95) | The range that plausibly contains the true value. A wide range means uncertainty. |
| API key | A password from a model provider that lets the app send requests on your account. |
