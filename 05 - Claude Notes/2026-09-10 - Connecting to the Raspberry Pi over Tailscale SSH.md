---
name: Connecting to the Raspberry Pi over Tailscale SSH
description: How to reach the Pi (raspberrypi @ 100.77.17.38) over Tailscale, and why SSH appears to hang silently on the first attempt of a session.
---

# Connecting to the Raspberry Pi over Tailscale SSH

Session date: 2026-09-10

## The setup

The Pi hosting the edge pipeline (`03 - ML/`) and the clinician dashboard
(`07 - Website/`) is reachable only over the tailnet:

- Tailscale name: `raspberrypi` (`raspberrypi.tail03bdf4.ts.net`)
- Tailscale IP: `100.77.17.38`
- Login user: `ranilo`
- Project path on the Pi: `/home/ranilo/Arrhythmia Thesis/`
- Dashboard: `http://100.77.17.38:8080/` (verified `200 OK` this session)

Port 22 on that address is **not** the Pi's own `sshd` — it is answered by
`tailscaled`. Reading the TCP banner directly returns `SSH-2.0-Tailscale`, which is
the quickest way to confirm you are talking to Tailscale SSH rather than OpenSSH.

## Why the first SSH of a session looks like it hangs

The tailnet policy has Tailscale SSH set to `"action": "check"` for this
user-to-node rule. In check mode every new session requires a one-time browser
re-authentication. The `ssh` client prints the challenge to **stderr** and then
blocks indefinitely waiting for the user to click it:

```
# Tailscale SSH requires an additional check.
# To authenticate, visit: https://login.tailscale.com/a/<one-time-token>
```

The trap: any tooling that redirects or pipes stderr will not surface that line
until the process exits, and the process never exits on its own. The symptom is a
completely silent hang with zero output — which looks like a network failure, a
sandbox blocking egress, or a missing SSH key, and is none of those.
`-o BatchMode=yes` does not help, because this is not a standard password or
keyboard-interactive prompt.

Note that key-based auth is irrelevant here. There is no `id_*` key for this host on
the Windows box, and none is needed — Tailscale SSH authenticates by tailnet
identity, not by an SSH keypair.

## How to connect

Interactively, just run it and click the URL when it appears:

```
ssh ranilo@100.77.17.38
```

Non-interactively (or from tooling that captures output), start the client with
stderr redirected to a file, read the auth URL out of that file, and leave the
process running while the URL is visited — the token stays valid until then:

```powershell
$err = "$env:TEMP\ts_ssh_err.txt"
$p = Start-Process "C:\WINDOWS\System32\OpenSSH\ssh.exe" `
  -ArgumentList '-n','-o','BatchMode=yes','ranilo@100.77.17.38','hostname' `
  -RedirectStandardError $err -RedirectStandardOutput "$env:TEMP\ts_ssh_out.txt" `
  -NoNewWindow -PassThru
Select-String -Path $err -Pattern 'https://login\.tailscale\.com/\S+'
```

Once the check passes, it is cached for the tailnet's `checkPeriod` (12 hours by
default), so subsequent commands in the same working session connect straight
through with no prompt. This was verified: a second command immediately afterwards
ran with no challenge.

## Diagnostic shortcuts

- `tailscale status` — confirms the peer is online and gives the IP.
- `Test-NetConnection 100.77.17.38 -Port 22` — proves reachability without
  involving SSH auth at all. This succeeds even while `ssh` is hanging, so it is
  useful for ruling out the network as the cause.
- Reading the raw TCP banner on port 22 distinguishes Tailscale SSH
  (`SSH-2.0-Tailscale`) from the Pi's own `sshd`.

## Open decision

The check-mode prompt could be removed by changing the tailnet policy at
`login.tailscale.com/admin/acls` from `"action": "check"` to `"action": "accept"`
for this user-to-node rule. That weakens the tailnet's security posture in exchange
for convenience, and at roughly one click per 12 hours the current setting is not
especially burdensome, so it has been left as `check`. This is the account owner's
call, not something to change on Claude's own initiative.

---
Back to [[Index|Claude Notes Index]]
