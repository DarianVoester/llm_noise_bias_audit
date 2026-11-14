# Disclaimer, liability and user responsibility

> Short version: This is a free, non-commercial hobby/research project by a private
> individual. It comes **without any warranty**. You decide which models, data, keys
> and code you use, and you are responsible for the consequences, including API
> costs, data protection, model licenses and any decision you base on the results.
>
> This document is not legal advice. If you need legal certainty for your own situation,
> ask a qualified lawyer.

## 1. Nature of the project (non-commercial)

- The software is developed and published by a private individual, free of charge,
  as free and open-source software under the [MIT License](LICENSE).
- It is supplied outside the course of any commercial activity: it is not sold, there
  is no paid version, no paid support, no advertising, no tracking and no monetisation of
  user data.
- The maintainer does not operate the software as a hosted service. Every user runs
  their own copy (GitHub Codespaces, Dev Container or local machine) and is the
  operator of that copy.
- Anyone who forks, redistributes, hosts or integrates this software as part of a
  commercial activity acts on their own responsibility and takes on any obligations that
  come with that (for example under the EU Cyber Resilience Act (EU) 2024/2847 or the EU
  Product Liability Directive (EU) 2024/2853). The original author is not the manufacturer
  of such derived products or services.

## 2. No warranty

The software is provided **"AS IS"**, without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular
purpose, correctness of results, availability, security and non-infringement (see
[LICENSE](LICENSE)). It is a research prototype: it may contain bugs, produce wrong
statistics, and change or stop being maintained at any time.

## 3. Limitation of liability

To the maximum extent permitted by applicable law, the author and contributors are not
liable for any damages or losses arising from the use of, or the inability to use, the
software, including but not limited to API charges, data loss, data leaks, lost profits,
wrong decisions or third-party claims.

Where applicable law does not allow a complete exclusion (for example German law, under
which free provision of software is generally treated as a gift, §§ 521, 523, 524 BGB),
the author is liable only for **intent and gross negligence** and for fraudulently
concealed defects. Nothing in this document limits liability that cannot be limited by
law.

## 4. Known limitations

Known defects and limits are documented openly so you can judge the risk yourself:

- Sandbox: the analysis-code sandbox is a best-effort defence (AST checks, separate
  process, audit hook, resource limits), not a hardened security boundary. A security
  review in November 2026 found and closed ways around the AST checks; more may exist.
  On Windows and macOS there are no memory limits. See [SECURITY.md](SECURITY.md).
- No login: the app has no user accounts or passwords. It listens on `127.0.0.1` only;
  whoever can reach it can use it with your keys.
- Endpoint checks: addresses of custom model servers are checked when you enter them, not
  again at the moment of each request (see "Port policy" in [SECURITY.md](SECURITY.md)).
- Statistics: results are estimates from a limited number of samples. Small `N`,
  parse failures or provider-side caching can distort them. Parse success rates are shown
  per model and variant; check them.
- Cost estimate: the cost/ETA guard uses approximate token counts and list prices.
  Real charges can be higher.
- Determinism: providers may ignore seeds or change models silently, so runs are not
  guaranteed to be reproducible.
- Answer parsing: the JSON/regex parsers can misread unusual model outputs.

## 5. Third-party components

This project depends on third-party open-source packages, listed with pinned versions and
hashes in [requirements.txt](requirements.txt). Each one is governed by its own license.
Model weights are not distributed with this project.

## 6. Your responsibilities as a user

By running the software you accept that you alone are responsible for:

| Area | Your responsibility |
|---|---|
| API keys & costs | Every request goes to the provider you choose, with your key, at your expense. Set spending limits with your provider. The built-in cost estimate is a best-effort guess, not a guarantee. |
| Provider terms | Complying with the terms of service and usage policies of OpenAI, Anthropic, Google, OpenRouter, Hugging Face, Ollama and any other service you connect. |
| Data you submit | Prompts and CSV files are sent to the model provider you select. **Do not submit personal data or confidential information** unless you have a legal basis and the necessary agreements. For any personal data you process, you are the controller under the GDPR, not the author of this software. |
| Model licenses | You download model weights yourself and must respect their licenses (e.g. Gemma Terms of Use, Llama Community License, Apache-2.0). |
| Analysis code | Python code you write or paste into the Statistics editor runs on your machine. Only run code you understand. |
| Port visibility | Keep port 8501 **private**. If you make it public, anyone with the URL can use your API keys and run analysis code. The same applies if you change the server address from `127.0.0.1` to `0.0.0.0` or publish the port from a container. |
| Custom endpoints | You choose which servers receive your prompts and keys. Allow public endpoints only for servers you trust, and give each server a key variable of its own. |
| Use of results | Results are statistical estimates of model behaviour for research and education. They are not professional, legal, medical, financial or HR advice. |

## 7. Prohibited and high-risk uses

Do **not** use this software, or its results, to make or support decisions about
individual people (for example hiring, credit, insurance, education, housing, law
enforcement or access to public services). Such uses can be high-risk under the EU AI
Act (EU) 2024/1689 and other laws, and they require safeguards this research prototype
does not provide. If you do it anyway, you do so entirely at your own risk and
responsibility.

## 8. Privacy: what this project collects

Nothing. The author does not operate any server for this project and receives no data
from your copy. Streamlit usage statistics are disabled in the Dev Container
(`STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`). API keys entered in the sidebar stay in
the browser session's memory. Network traffic only goes to services you configure
(model providers, Ollama / Hugging Face downloads, package indexes). Those services
process your data under their own privacy policies. GitHub's privacy statement applies
to Codespaces and the repository.

## 9. Third parties & trademarks

This project is not affiliated with, endorsed by or sponsored by OpenAI, Anthropic,
Google, OpenRouter, Meta, Ollama, Hugging Face, GitHub, Streamlit, or the authors and
publishers of *Noise: A Flaw in Human Judgment*. All product names and trademarks belong to
their respective owners and are used only to describe compatibility.

## 10. AI-assisted development

Large parts of the specification and code were generated with AI assistants (see the
README). The code has **not** necessarily been reviewed line by line by a human, and the
automated tests cover only selected cases. Merging a change, passing CI, or any statement in
this repository (README, SPEC, SECURITY, CONTRIBUTING, code comments, commit messages, UI
texts) is **not** a confirmation that the code is correct, complete, secure or fit for any
purpose, and it is **not** an assumption of responsibility or liability by the author.

## 11. How to read statements in this repository

Descriptions of features and security measures (for example "keys are not stored",
"sandbox", "loopback only", "verified", "reproducible", "blocked") describe design goals
and intended behaviour. They are not warranties, guarantees or promised characteristics.
Words such as "must", "never" or "always" in the documentation are instructions or design
intentions, not promises. Response times, maintenance and security fixes are voluntary and
can stop at any time. If any statement in this repository seems to conflict with this
disclaimer or the [LICENSE](LICENSE), this disclaimer and the LICENSE take precedence.

## 12. Contact

Questions or problems: open an issue in the GitHub repository. For security
vulnerabilities, use private vulnerability reporting (see [SECURITY.md](SECURITY.md)).
