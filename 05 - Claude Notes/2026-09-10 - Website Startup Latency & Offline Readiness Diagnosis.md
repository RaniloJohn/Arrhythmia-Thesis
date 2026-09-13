---
title: Website Startup Latency & Offline Readiness Diagnosis
tags: [website, performance, offline, iso25010, diagnosis]
date: 2026-09-10
status: diagnosed — fix delegated to Antigravity via PLAN.md
reviewer: Claude (Research & Theoretical Architect)
---

# Website Startup Latency & Offline Readiness Diagnosis

Investigation of the user report that "the localhost takes so long to run the
website." Measured on both tiers (Windows dev host and the Raspberry Pi
deployment target) on 2026-09-10.

## Summary of the finding

The server is not slow. The **first paint of the page is blocked for up to 21
seconds by a render-blocking Google Fonts stylesheet** in `<head>` of
`07 - Website/frontend/index.html`. The stall appears only when the browser's
network cannot reach `fonts.googleapis.com` — which is precisely the intended
deployment condition for this thesis (offline barangay health centers). On a
machine with working internet the same page loads in well under a second, which
is why the problem has appeared intermittent and hard to pin down.

## Measurements

Backend module load, profiled by requiring each dependency in sequence:

| Stage | Time |
|---|---|
| `require('express')` | 137 ms |
| `require('jsonwebtoken')` | 55 ms |
| `require('ws')` | 12 ms |
| `require('cors')`, `bcryptjs`, `config`, `db`, `pubsub_relay` | 12 ms combined |
| **Total backend cold start** | **216 ms** |

An earlier figure of ~2.9 s for "startup to ACTIVE" was an artifact of the
100 ms polling interval and process-launch overhead in the measurement harness,
not real server latency. It should not be quoted as a result.

Asset delivery, measured with `curl` against the running server:

| Asset | On the Pi (localhost) | From Windows host |
|---|---|---|
| `index.html` (31,209 B) | 2–4 ms | 351 ms cold, then ~15 ms |
| `css/styles.css` (20,651 B) | 6.7 ms | 15 ms |
| `js/app.js` | — | 8 ms |
| `/api/health` | — | 12 ms |

Every locally served asset is fast. The external dependency is the outlier:

| Condition | Blocking time |
|---|---|
| `fonts.googleapis.com` reachable | 134 ms |
| `fonts.googleapis.com` unreachable (blackhole route, simulating an offline site) | **21.1 s** before timeout |

The 21.1 s figure was produced by forcing resolution of `fonts.googleapis.com`
to a non-routable address and timing the connection attempt. A browser
encountering the same condition blocks first paint on the stylesheet, so the
clinician sees a blank white page for that entire period.

## Mechanism

`index.html` lines 7–9 declare:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:...&family=Public+Sans:..." rel="stylesheet">
```

A stylesheet `<link>` in `<head>` is render-blocking by specification: the
browser will not paint until it is fetched or the request fails. `css/styles.css`
then sets `--font-family: 'Public Sans', ...` and `--font-heading: 'Newsreader',
...`, so the fonts are genuinely referenced rather than vestigial — but both
declarations already carry complete local fallback stacks (`-apple-system`,
`Segoe UI`, `system-ui`, `sans-serif`), meaning the page is fully legible
without the remote fonts.

## Why this matters beyond speed

This is not only a performance defect; it contradicts a load-bearing claim of
the thesis. `ANTIGRAVITY.md` §1 states the system delivers "100% offline-capable"
operation, and Chapter 1's research questions frame the device for primary-care
and non-hospital settings without reliable connectivity. A dashboard whose first
paint depends on a Google CDN is not offline-capable. Under ISO/IEC 25010 this
degrades two of the six characteristics being evaluated:

- **Performance Efficiency** — the sensor-to-browser budget in `config.js` is
  150 ms; a 21 s first paint exceeds it by two orders of magnitude.
- **Reliability** — "graceful offline fallback" is a stated reliability target,
  and the current behaviour is a blank page rather than a degraded one.

It is also worth recording that the Pi *does* currently have internet access
(`fonts.googleapis.com` connects in 65 ms from the Pi), which is why the live
deployment has not exhibited the stall. The defect is latent, and would surface
exactly when the system is demonstrated in its intended offline setting.

## Recommended fix

Self-host the two font families in `07 - Website/frontend/assets/fonts/` and
serve them from the Node.js static handler, removing all three external `<link>`
elements. This preserves the existing visual design while making first paint
independent of any network beyond localhost. The `@font-face` declarations should
use `font-display: swap` so text renders immediately from the fallback stack even
while a local font file is still being read.

Rejected alternatives: making the stylesheet non-blocking via
`media="print" onload="this.media='all'"` still leaves a failed network request
and a font swap on every load; dropping the custom fonts entirely would change
the approved visual design unnecessarily.

Implementation is delegated to Antigravity — see `PLAN.md` §1.

## Related

- [[2026-09-10 - Connecting to the Raspberry Pi over Tailscale SSH|Tailscale SSH connection notes]]
- [[../02 - Code Review/2026-09-10 - Seeded Demo Data, Dead Sensor Link & Offline Blocking Assets|Code review: seeded demo data & dead sensor link]]

---
Back to [[Index|Claude Notes Index]]

---

## Addendum (same day): the user's actual blocker was not this

The analysis above is correct about the offline font dependency, but it was **not**
the reason the user could not open the dashboard. That was diagnosed separately and
the finding supersedes the framing above.

The user browses the dashboard in **Chromium on the Pi's own desktop**. In that
browser the page hangs indefinitely on a blank white tab reading "Loading...".
Evidence gathered over Tailscale SSH:

- The document request `http://localhost:8080/` never completes. A Chrome DevTools
  Protocol network trace showed it still `PENDING` after 25 s, with no other request
  even attempted — so fonts, JavaScript, the canvas and the WebSocket are all
  irrelevant to this failure; it dies before any of them.
- `ss` reports **zero established connections** to port 8080 while Chromium is
  loading. The TCP handshake never completes.
- The same page fetched with `curl` on the same machine at the same moment returns
  HTTP 200 in **1.5 ms**, including with a full browser-shaped header set.
- The failure reproduces in a **fresh Chromium profile**, so it is not profile
  corruption, cache, HSTS or a service worker.
- Chromium also fails to load `http://example.com/` and `https://duckduckgo.com/`,
  while `curl` reaches example.com in 193 ms. It fails equally with `--no-sandbox`
  and with the network service run in-process.
- **Firefox on the same Pi renders the dashboard correctly**, fully styled, from
  `http://localhost:8080/`.

The conclusion is that Chromium's networking on this Pi is broken for every
destination, and the dashboard is simply one of its victims. The web application,
the Node.js server and the network path are all healthy. Ruled out along the way:
IPv4/IPv6 binding (the server listens dual-stack on `*:8080` and `curl` succeeds over
both `::1` and `127.0.0.1`), an HTTP-to-HTTPS upgrade (a TLS request to the port fails
in 3.6 ms rather than hanging), proxy configuration, DNS (plain-IP URLs fail
identically), and an upgrade-while-running (the package was upgraded 2026-08-26, the
browser started 2026-09-10).

Immediate workaround: use Firefox on the Pi, or browse to the Pi from another machine
on the tailnet — from the Windows host the dashboard loads in 217 ms.

Repairing Chromium itself is a system-administration task on the Pi, unrelated to this
codebase, and has not been attempted.

### Measurements worth keeping from this investigation

- Backend cold start: 216 ms. Every static asset on the Pi: 1.4–6.7 ms.
- Login (`POST /api/auth/login`): 436–503 ms, spent almost entirely in
  `bcrypt.compareSync` at cost 12 using **bcryptjs**, the pure-JavaScript
  implementation. Because `compareSync` is synchronous it blocks the Node.js event
  loop for its whole duration — `index.html` served during a login goes from 1.4 ms to
  380 ms. This is a real defect independent of the Chromium problem and should be
  fixed by switching to the async `bcrypt.compare`.
- Pi hardware: Raspberry Pi 5, 8 GB, not thermally throttled, 50 °C, no swap in use.
