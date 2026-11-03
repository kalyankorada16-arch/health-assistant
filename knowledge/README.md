# Approved reference documents

Only files in this folder can ground answers. Add one `.md` or `.txt` file per source,
starting with this front-matter block (all fields required, plus at least one date):

```
---
title: Sore throat
publisher: <organisation that published it>
url: https://<the page you copied it from>
published: 2024-05-01
reviewed: 2025-01-15
version: <edition, page version or retrieval date>
---
<the text of the reference, copied or summarised faithfully>
```

Rules:
- Use only trustworthy public-health sources (for example WHO, NHS, CDC, Indian government health bodies) and respect each site's licence and reuse terms.
- Keep the text faithful to the source. Do not add your own medical claims.
- A file with missing metadata is skipped and the app shows a warning.
- This README and files that do not end in `.md` or `.txt` are ignored.

Current documents: six NHS pages (cough, fever in adults, diarrhoea and vomiting, headaches, common cold, and sore throat from NHS 111 Wales), each a faithful own-words summary of the page with its review date and retrieval date. Please compare each against its source URL before final submission.

Limitations of the current set:
- The sources are UK pages, so they mention UK services (NHS 111, 999, GP). The app's emergency numbers are for India. Consider adding Indian sources (for example Ministry of Health and Family Welfare or ICMR guidance) so the library matches the deployment region.
- Only these six topics are covered. Questions about anything else correctly return "not covered".
- The NHS cough and diarrhoea pages are due for review in December 2026, so re-check them then.
